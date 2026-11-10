// Apple on-device dictation (macOS 26 Speech framework: SpeechAnalyzer + DictationTranscriber),
// streamed in real time, logging what is on screen against the audio clock.
// Each clip is fed in 80 ms chunks at real-time pace; whenever the transcriber reports a result,
// the text shown (finalized text + current volatile text) is logged with the seconds of audio fed so far.
// Output: one JSON line per clip, {id, dur, partials: [[t, text], ...], final, step_ms: []} (Earlybird's log format).
// usage: swift apple_dictation.swift manifest.jsonl > apple-dictation.jsonl
import AVFoundation
import Foundation
import Speech

let SR = 16000.0, CHUNK = 1280   // 80 ms

func err(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

struct Clip { let id: String; let dur: Double; let path: String }

func loadManifest(_ path: String) throws -> [Clip] {
    let dir = (path as NSString).deletingLastPathComponent
    return try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n").map { line in
        let o = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
        return Clip(id: o["id"] as! String, dur: o["dur"] as! Double, path: (dir as NSString).appendingPathComponent(o["audio"] as! String))
    }
}

final class Screen: @unchecked Sendable {
    private let lock = NSLock()
    var finalized = "", volatile = "", fed = 0.0
    var partials: [(Double, String)] = []
    func text() -> String {
        (finalized + " " + volatile).split(whereSeparator: { $0 == " " || $0 == "\n" }).joined(separator: " ")
    }
    func update(_ s: String, isFinal: Bool) {
        lock.lock(); defer { lock.unlock() }
        if isFinal { finalized += " " + s; volatile = "" } else { volatile = s }
        let t = text()
        if partials.last?.1 != t { partials.append((fed, t)) }
    }
    func setFed(_ t: Double) { lock.lock(); fed = t; lock.unlock() }
}

func run(_ clip: Clip, locale: Locale) async throws -> [String: Any] {
    let transcriber = DictationTranscriber(locale: locale, contentHints: [], transcriptionOptions: [],
                                           reportingOptions: [.volatileResults, .frequentFinalization], attributeOptions: [])
    if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        err("downloading Apple's dictation model (once)…"); try await req.downloadAndInstall()
    }
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    guard let target = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else { throw NSError(domain: "no audio format", code: 1) }

    let file = try AVAudioFile(forReading: URL(fileURLWithPath: clip.path))
    let src = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: SR, channels: 1, interleaved: false)!
    let whole = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))!
    try file.read(into: whole)
    guard file.processingFormat.sampleRate == SR, file.processingFormat.channelCount == 1 else { throw NSError(domain: "expected 16 kHz mono", code: 2) }
    let converter = AVAudioConverter(from: src, to: target)!

    let screen = Screen()
    let (stream, cont) = AsyncStream<AnalyzerInput>.makeStream()
    try await analyzer.start(inputSequence: stream)
    let reader = Task {
        for try await r in transcriber.results { screen.update(String(r.text.characters), isFinal: r.isFinal) }
    }
    let total = Int(whole.frameLength), samples = whole.floatChannelData![0]
    var pos = 0
    let clock = ContinuousClock(), start = clock.now
    while pos < total {
        let n = min(CHUNK, total - pos)
        let chunk = AVAudioPCMBuffer(pcmFormat: src, frameCapacity: AVAudioFrameCount(n))!
        chunk.frameLength = AVAudioFrameCount(n)
        memcpy(chunk.floatChannelData![0], samples + pos, n * MemoryLayout<Float>.size)
        let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(n) * target.sampleRate / SR) + 64)!
        var given = false; var e: NSError?
        converter.convert(to: out, error: &e) { _, status in
            if given { status.pointee = .noDataNow; return nil }
            given = true; status.pointee = .haveData; return chunk
        }
        if let e { throw e }
        pos += n
        screen.setFed(Double(pos) / SR)
        cont.yield(AnalyzerInput(buffer: out))
        try await clock.sleep(until: start + .milliseconds(Int(Double(pos) / SR * 1000)))   // real-time pace
    }
    cont.finish()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    _ = try await reader.value
    return ["id": clip.id, "dur": clip.dur, "partials": screen.partials.map { [$0.0, $0.1] },
            "final": screen.text(), "step_ms": [Double]()]
}

let args = CommandLine.arguments
guard args.count > 1 else { err("usage: swift apple_dictation.swift manifest.jsonl > out.jsonl"); exit(2) }
guard let locale = await DictationTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US")) else { err("en-US dictation is not supported on this Mac"); exit(1) }
let clips = try loadManifest(args[1])
for (k, c) in clips.enumerated() {
    let o = try await run(c, locale: locale)
    let data = try JSONSerialization.data(withJSONObject: o)
    print(String(data: data, encoding: .utf8)!); fflush(stdout)
    err("\(k + 1)/\(clips.count) \(c.id): \(o["final"] as! String)")
}
