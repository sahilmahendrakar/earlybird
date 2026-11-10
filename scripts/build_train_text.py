"""Text for the training corpus: WikiText-103 train sentences plus short spoken commands.

The commands come from templates with slots, so they are varied in wording but stay in
the plain-letters form the rest of the pipeline expects (numbers are spelled out).
usage: build_train_text.py OUT.txt [N_WIKI] [N_COMMANDS]
Also writes the commands alone to OUT.commands.txt."""
import os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
from text import wikitext_sentences

NUM = "one two three four five six seven eight nine ten twelve fifteen twenty thirty fifty ninety".split() + ["forty five"]
NAMES = "Maria James Priya Wei Sarah David Fatima Carlos Emma Noah Aisha Liam Sofia Ethan Mei Omar Hannah Lucas Grace Daniel".split() \
    + ["my mom", "my dad", "my sister", "my brother", "my boss", "the team"]
APPS = ["the calendar", "the camera", "my email", "the browser", "settings", "the calculator", "maps", "my notes",
        "the music app", "the weather app", "my photos", "the terminal", "the file manager", "messages"]
ROOMS = ["kitchen", "living room", "bedroom", "office", "hallway", "bathroom", "garage", "basement"]
DEVICES = ["lights", "fan", "heater", "speaker", "television", "thermostat", "lamp", "air conditioner"]
PLACES = ["the airport", "the nearest gas station", "work", "home", "the grocery store", "the train station",
          "the closest pharmacy", "downtown", "the library", "the gym", "a coffee shop nearby", "the hospital"]
MUSIC = ["some jazz", "my workout playlist", "classical music", "the next song", "something relaxing", "the latest episode",
         "that song again", "the morning news", "rock music", "my favorites", "an audiobook", "some piano music"]
DAYS = ["today", "tomorrow", "tonight", "this weekend", "on Monday", "on Friday", "next week", "this afternoon", "tomorrow morning"]
TIMES = ["at six", "at seven thirty", "at noon", "at nine in the morning", "at half past eight", "in the evening", "at midnight", "at five o'clock"]
UNITS = ["seconds", "minutes", "hours"]
THINGS = ["buy milk", "call the dentist", "water the plants", "pay the rent", "take out the trash", "send the report",
          "pick up the kids", "book a flight", "charge the car", "feed the cat", "renew my passport", "check the oven"]
MSGS = ["I am running late", "I will be there soon", "can you call me back", "the meeting is moved to three",
        "thanks for your help", "see you tonight", "I just left the office", "do we need anything from the store",
        "happy birthday", "let me know when you arrive", "sounds good to me", "I am on my way"]
EDITS = ["Delete the last sentence", "New paragraph", "Undo that", "Select the whole line", "Capitalize that word",
         "Go to the end of the document", "Scratch that", "Replace it with the previous version", "Copy this and paste it below",
         "Make that a bullet list", "Read that back to me", "Stop listening", "Start dictation", "Insert a comma there",
         "Go back one page", "Scroll down", "Scroll up a little", "Close this window", "Save the file", "Open a new tab",
         "Switch to the other window", "Zoom in", "Zoom out", "Take a screenshot", "Search for that", "Cancel", "Never mind",
         "Yes", "No", "Go ahead", "Try again", "Stop", "Pause", "Continue", "Next", "Go back", "Send it", "What time is it",
         "What is the date today", "How is the weather", "Mute the microphone", "Turn the volume up", "Turn the volume down"]
POLITE = ["", "", "", "Please ", "Hey, ", "Okay, ", "Can you ", "Could you "]

T = [
    lambda r: f"set a timer for {r.choice(NUM)} {r.choice(UNITS)}",
    lambda r: f"set an alarm {r.choice(TIMES)} {r.choice(DAYS)}",
    lambda r: f"wake me up {r.choice(TIMES)}",
    lambda r: f"remind me to {r.choice(THINGS)} {r.choice(DAYS)}",
    lambda r: f"remind me to {r.choice(THINGS)} in {r.choice(NUM)} {r.choice(UNITS)}",
    lambda r: f"add {r.choice(THINGS)} to my list",
    lambda r: f"open {r.choice(APPS)}",
    lambda r: f"close {r.choice(APPS)}",
    lambda r: f"turn on the {r.choice(ROOMS)} {r.choice(DEVICES)}",
    lambda r: f"turn off the {r.choice(DEVICES)} in the {r.choice(ROOMS)}",
    lambda r: f"set the {r.choice(ROOMS)} thermostat to {r.choice(['sixty eight', 'seventy', 'seventy two', 'sixty five'])} degrees",
    lambda r: f"dim the {r.choice(ROOMS)} lights to {r.choice(['ten', 'twenty', 'thirty', 'fifty', 'eighty'])} percent",
    lambda r: f"play {r.choice(MUSIC)}",
    lambda r: f"play {r.choice(MUSIC)} in the {r.choice(ROOMS)}",
    lambda r: f"skip ahead {r.choice(NUM)} {r.choice(UNITS)}",
    lambda r: f"call {r.choice(NAMES)}",
    lambda r: f"send a message to {r.choice(NAMES)} saying {r.choice(MSGS)}",
    lambda r: f"text {r.choice(NAMES)} that {r.choice(MSGS)}",
    lambda r: f"reply {r.choice(MSGS)}",
    lambda r: f"email {r.choice(NAMES)} about {r.choice(['the budget', 'the schedule', 'the trip', 'the contract', 'the launch', 'lunch'])}",
    lambda r: f"schedule a meeting with {r.choice(NAMES)} {r.choice(DAYS)} {r.choice(TIMES)}",
    lambda r: f"what is on my calendar {r.choice(DAYS)}",
    lambda r: f"navigate to {r.choice(PLACES)}",
    lambda r: f"how long does it take to get to {r.choice(PLACES)}",
    lambda r: f"find {r.choice(PLACES)}",
    lambda r: f"what is the weather like {r.choice(DAYS)}",
    lambda r: f"will it rain {r.choice(DAYS)}",
    lambda r: f"search for {r.choice(['cheap flights', 'a good recipe for dinner', 'the nearest hardware store', 'how to tie a tie', 'the score of the game', 'movie times tonight'])}",
    lambda r: f"what is {r.choice(NUM)} times {r.choice(NUM)}",
    lambda r: f"how many {r.choice(['ounces are in a pound', 'days until the weekend', 'miles is ten kilometers', 'calories are in an apple'])}",
    lambda r: f"take a note, {r.choice(MSGS)}",
]


def command(r):
    if r.random() < 0.2:
        return r.choice(EDITS) + "."
    body = r.choice(T)(r)
    question = body.split()[0] in ("what", "how", "will")
    pre = r.choice(["", "", "Hey, ", "Okay, "] if question else POLITE)
    ask = question or pre in ("Can you ", "Could you ")
    s = pre + body
    return s[0].upper() + s[1:] + ("?" if ask else ".")


def main():
    out = sys.argv[1]
    n_wiki = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
    n_cmd = int(sys.argv[3]) if len(sys.argv) > 3 else 15000
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    r = random.Random(0)
    cmds = set()
    for _ in range(n_cmd * 20):
        if len(cmds) >= n_cmd:
            break
        cmds.add(command(r))
    open(os.path.splitext(out)[0] + ".commands.txt", "w").write("\n".join(sorted(cmds)) + "\n")
    wiki = []
    for s in wikitext_sentences("train", min_words=4, max_words=28):
        wiki.append(s)
        if len(wiki) >= n_wiki:
            break
    lines = wiki + sorted(cmds)
    r.shuffle(lines)
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"{len(wiki)} wiki + {len(cmds)} commands -> {out}")


if __name__ == "__main__":
    main()
