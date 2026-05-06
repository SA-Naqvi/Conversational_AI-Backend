import json

deltas = [
    "<think>", "\n", "Okay", ",", " the", " user", " sent", " \"", "hi", "\".", " That", "'s", " a", " greeting", ".",
    "</think>", "\nHello", "!"
]

def stream_filter(deltas):
    buffer = ""
    in_think_block = False
    
    for delta in deltas:
        buffer += delta
        while True:
            if in_think_block:
                end_idx = buffer.find("</think>")
                if end_idx != -1:
                    in_think_block = False
                    buffer = buffer[end_idx + 8:]
                    continue
                else:
                    if len(buffer) > 8:
                        buffer = buffer[-8:]
                    break
            else:
                start_idx = buffer.find("<think>")
                if start_idx != -1:
                    if start_idx > 0:
                        print("Yielding:", repr(buffer[:start_idx]))
                    in_think_block = True
                    buffer = buffer[start_idx + 7:]
                    continue
                else:
                    if len(buffer) > 7:
                        y = buffer[:-7]
                        print("Yielding:", repr(y))
                        buffer = buffer[-7:]
                    break
                    
    if buffer and not in_think_block and "<think" not in buffer:
        print("Yielding:", repr(buffer))

stream_filter(deltas)
