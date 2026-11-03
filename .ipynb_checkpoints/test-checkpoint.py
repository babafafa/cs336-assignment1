import re
text = "abc<|endoftext|>def"
parts = re.split(r"<endoftext>", text)
print(parts)   # ['abc<|endoftext|>def']  ← 没切开！