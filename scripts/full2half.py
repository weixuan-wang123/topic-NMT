# -*- coding: utf-8 -*-  

import sys
import six
import codecs
import unicodedata

FULL2HALF = dict((i + 0xFEE0, i) for i in range(0x21, 0x7F))
FULL2HALF[0x3000] = 0x20

HALF2FULL = dict((i, i + 0xFEE0) for i in range(0x21, 0x7F))
HALF2FULL[0x20] = 0x3000

def halfen(s):
  '''
  Convert full-width characters to ASCII counterpart
  '''
  return str(s).translate(FULL2HALF)


def fullen(s):
  '''
  Convert all ASCII characters to the full-width counterpart.
  '''
  return str(s).translate(HALF2FULL)


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def clean_text(text):
  """Performs invalid character removal and whitespace cleanup on text."""
  output = []
  for char in text:
    cp = ord(char)
    if cp == 0 or cp == 0xfffd or _is_control(char):
      continue
    if _is_whitespace(char):
      output.append(" ")
    else:
      output.append(char)
  return "".join(output)


sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf-8', buffering=True)
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=True)

#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

#for line in codecs.getreader("utf-8")(sys.stdin):
for line in sys.stdin:
  line = line.strip()
  line = halfen(line.strip())
  line = convert_to_unicode(line)
  line = clean_text(line)
  line = " ".join(line.strip().split())
  print(line)
