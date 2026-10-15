"""Independent, hand-authored case writer. Never imports project scripts or data."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = []
CATEGORY = None


def C(description, pattern, yes, no, pitfall, flags="", contains=False):
    case(description, pattern, yes.split("|"), no.split("|"), pitfall, flags, contains)


def fixed(description, prefix, alphabet, width, suffix="", flags=""):
    """Fixed-width test samples constructed from the stated alphabet, not the regex."""
    import re
    a, b, c = alphabet[0], alphabet[-1], alphabet[len(alphabet)//2]
    samples = [a*width, b*width, c*width, (a+b)*width, (b+a)*width, a+b*(width-1), b+a*(width-1), c+a*(width-1)]
    yes = [prefix+x[:width]+suffix for x in samples]
    if len(set(yes)) < 8:
        from itertools import product
        for sequence in product(alphabet, repeat=width):
            candidate = prefix+"".join(sequence)+suffix
            if candidate not in yes:
                yes.append(candidate)
            if len(set(yes)) >= 8:
                break
    no = [prefix+a*(width-1)+suffix, prefix+a*(width+1)+suffix,
          prefix+"?"+a*(width-1)+suffix, prefix+" "+a*(width-1)+suffix,
          prefix+a*(width-1)+"é"+suffix, prefix+a*width+"\t"+suffix]
    if prefix:
        no.append(prefix[:-1]+"?"+a*width+suffix)
    if suffix:
        no.append(prefix+a*width+"?"+suffix[1:])
    for char in alphabet:
        if char.swapcase() != char and char.swapcase() not in alphabet:
            no.append(prefix+char.swapcase()+a*(width-1)+suffix)
            break
    escaped = lambda x: re.escape(x).replace("/", r"\/")
    case(description, escaped(prefix)+"["+alphabet+"]{"+str(width)+"}"+escaped(suffix), yes, no,
         "Literal punctuation, allowed characters, and exact width all matter.", flags)


def bounded(description, prefix, low, high, pattern, suffix=""):
    import re
    values = [low, low+1, low+2, high-2, high-1, high, (low+high)//2, (low+high)//2+1]
    yes = [prefix+str(v)+suffix for v in values]
    no = [prefix+str(low-1)+suffix, prefix+str(high+1)+suffix, prefix+"0"+str(low)+suffix,
          prefix+"+"+str(low)+suffix, prefix+str(low)+".0"+suffix, prefix+" "+str(low)+suffix,
          prefix+str(high)+"0"+suffix]
    esc = lambda x: re.escape(x).replace("/", r"\/")
    case(description, esc(prefix)+pattern+esc(suffix), yes, no, "Endpoints and canonical decimal spelling must both be enforced.")


def search(description, prefix, token_pattern, tokens, suffix, bad_tokens):
    import re
    esc = lambda x: re.escape(x).replace("/", r"\/")
    yes = [prefix+t+suffix for t in tokens[:4]] + ["before "+prefix+t+suffix+" after" for t in tokens[4:8]]
    no = [prefix+t+suffix for t in bad_tokens] + [prefix.upper()+tokens[0]+suffix, prefix+tokens[0]+" "+suffix]
    case(description, "(?<="+esc(prefix)+")"+token_pattern+"(?="+esc(suffix)+")", yes, no,
         "Search anywhere, but require the exact local context and the entire bounded token.", contains=True)


def equal_pair(description, prefix, sep, token_pattern, tokens):
    import re
    esc = lambda x: re.escape(x).replace("/", r"\/")
    yes = [prefix+t+sep+t for t in tokens]
    first = tokens[0]
    no = [prefix+first+sep+tokens[1], prefix+first+sep+first+first,
          prefix+"?"+sep+"?", prefix+first+" "+sep+first,
          prefix+first+sep+" "+first, prefix+first+sep+first+sep+first,
          prefix+"!"*len(first)+sep+"!"*len(first)]
    for bad in [first[:-1], first+"0", re.sub(r"[0-9]", "z", first), re.sub(r"[A-Za-z]", "0", first)]:
        if bad != first:
            no.append(prefix+bad+sep+bad)
    # Confirmation imports preserve the quote convention as well as the value.
    quoted_yes = []
    for index, token in enumerate(tokens):
        quote = "'" if index % 2 == 0 else '"'
        quoted_yes.append(prefix+quote+token+quote+sep+quote+token+quote)
    quoted_no = []
    for text in no:
        body = text[len(prefix):]
        if sep in body:
            left, right = body.split(sep, 1)
            quoted_no.append(prefix+"'"+left+"'"+sep+"'"+right+"'")
    quoted_no += [prefix+"'"+first+'"'+sep+"'"+first+"'",
                  prefix+"'"+first+"'"+sep+'"'+first+'"', prefix+first+sep+first]
    description += " Each of the two values must be enclosed in matching ASCII single or double quotes; the same quote character must be used for both values. Quotes are not part of the value, and escapes are not permitted."
    # Existing payload patterns use no capturing groups. Rewrite an end-or-separator
    # lookahead in the one policy-secret case to see the closing quote instead.
    token_pattern = token_pattern.replace(r"(?:\/|$)", r"['\"]")
    case(description, esc(prefix)+r"(['\"])("+token_pattern+r")\1"+esc(sep)+r"\1\2\1", quoted_yes, quoted_no,
         "Quote style, both closing quotes, value policy and exact value repetition must all agree.")


def envelope(description, opening, closing, marker):
    import re
    esc = lambda x: re.escape(x).replace("/", r"\/")
    interior = [marker, "a"+marker, marker+"b", "\n"+marker, marker+"\n", "\t"+marker,
                "😀"+marker, "x"*(40-len(marker))+marker, "😀"*(40-len(marker))+marker]
    labels = ["aa", "abcdef", "tag", "route", "memo", "ab", "zz", "key", "xy"]
    yes = [opening+label+":"+body+":"+label+closing for label, body in zip(labels, interior)]
    wrap = lambda body: opening+"aa:"+body+":aa"+closing
    no = [wrap(""), wrap("abc"), wrap("x"*(41-len(marker))+marker),
          opening+"aa:"+marker+":ab"+closing,
          opening+"a:"+marker+":a"+closing,
          opening+"abcdefg:"+marker+":abcdefg"+closing,
          opening+"AA:"+marker+":AA"+closing, wrap(marker.upper()),
          opening+"aa:"+marker+":aa", "aa:"+marker+":aa"+closing,
          wrap("😀"*(41-len(marker))+marker)]
    subject = description.split(" is ",1)[0].split(" must ",1)[0]
    description = subject + ": the entire string has the literal opening wrapper " + opening + ", a lowercase ASCII routing label of 2 through 6 letters, colon, a body, colon, the exact same routing label and the literal closing wrapper " + closing + ". The body has 1 through 40 Unicode code points, allows any characters including line breaks and colons, and must contain the lowercase substring " + marker + ". Only the body counts toward its length limit; no other text is allowed."
    no += [opening+marker+":abc:"+marker+closing] if 2 <= len(marker) <= 6 else []
    yes += [opening+"aa:prefix: " + marker + ": suffix:aa"+closing]
    case(description, esc(opening)+r"(?<route>[a-z]{2,6}):(?=.{1,40}:\k<route>"+esc(closing)+r"$)(?=.{0,40}"+esc(marker)+r".{0,40}:\k<route>"+esc(closing)+r"$).*:\k<route>"+esc(closing), yes, no,
         "Repeat the routing label exactly, count only body code points including line breaks, and require the marker inside that body.", "su")


def unicode_name(description, prefix="", suffix=""):
    import re
    esc = re.escape(prefix).replace("/", r"\/")
    yes = [prefix+x+suffix for x in ["ab", "éx", "á̈b", "東京-大阪", "Δέλτα", "مرحبا", "Зима", "abcdefghijkl", "é-ë", "Aé"]]
    no = [prefix+x+suffix for x in ["a", "abcdefghijklm", "ab1", "a_b", "a b", "é", "😀a", "́ab", "á̈̃b", "-ab", "ab-", "a--b", "ab-cd-ef"]]
    topic = description.split(" is ", 1)[0].split(" consists ", 1)[0]
    description = topic + ": " + ("the literal prefix " + prefix + " followed by " if prefix else "") + "2 through 12 Unicode letter clusters. Each cluster is a Unicode letter followed by zero to two Unicode combining marks. One optional ASCII hyphen may join two nonempty runs of clusters, but no edge hyphen or second hyphen is allowed. Length counts letter clusters, excluding the hyphen and any fixed prefix; spaces, digits and other characters are forbidden. No normalization is performed."
    if suffix:
        rendered_suffix = suffix.replace("\n", "LF followed by ")
        description += " Append the exact literal suffix " + rendered_suffix + " after the name; its characters do not count toward the cluster limit."
        no += [prefix+"ab", prefix+"ab?"+suffix[1:]]
        if suffix.upper() != suffix:
            no.append(prefix+"ab"+suffix.upper())
    cluster = r"\p{L}\p{M}{0,2}"
    tail = re.escape(suffix).replace("/", r"\/").replace("\\\n", r"\n")
    case(description, esc+r"(?=(?:"+cluster+r"-?){2,12}"+tail+r"$)(?:"+cluster+r"){1,12}(?:-(?:"+cluster+r"){1,12})?"+tail, yes, no,
         "Count base-letter clusters, attach bounded marks only after a letter, and limit the optional internal separator.", "u")


def line_copy(description, prefix, token_pattern, tokens):
    import re
    esc = re.escape(prefix).replace("/", r"\/")
    yes = [prefix+t+"\ncopy="+t for t in tokens]
    t = tokens[0]
    no = [prefix+t+"\ncopy="+tokens[1], prefix+t+"\r\ncopy="+t,
          prefix+t+" copy="+t, prefix+t+"\nCopy="+t,
          prefix+"?\ncopy=?", prefix+t+"\ncopy="+t+" ",
          prefix+"!"*len(t)+"\ncopy="+"!"*len(t)]
    for bad in [t[:-1], t+"0", re.sub(r"[0-9]", "z", t), re.sub(r"[A-Za-z]", "0", t)]:
        if bad != t:
            no.append(prefix+bad+"\ncopy="+bad)
    quoted_yes = []
    for index, token in enumerate(tokens):
        quote = "'" if index % 2 == 0 else '"'
        quoted_yes.append(prefix+quote+token+quote+"\ncopy="+quote+token+quote)
    quoted_no = []
    for text in no:
        if "\ncopy=" in text:
            left, right = text[len(prefix):].split("\ncopy=", 1)
            quoted_no.append(prefix+"'"+left+"'\ncopy='"+right+"'")
        else:
            quoted_no.append(text)
    quoted_no += [prefix+"'"+t+'"'+"\ncopy='"+t+"'",
                  prefix+"'"+t+"'\ncopy=\""+t+'"', prefix+t+"\ncopy="+t,
                  prefix+"'"+t+"'\r\ncopy='"+t+"'"]
    description = description.replace("No final LF or whitespace", "No final LF or extra whitespace").replace("punctuation or spaces", "additional punctuation or spaces")
    description += " Both values must be enclosed in matching ASCII single or double quotes, using the same quote character on both lines. The quotes are delimiters, not part of the value; no escapes are permitted."
    token_pattern = token_pattern.replace(r"(?:\n|$)", r"['\"]")
    case(description, esc+r"(['\"])("+token_pattern+r")\1\ncopy=\1\2\1", quoted_yes, quoted_no,
         "The exact value and quote convention must agree across LF-separated lines, in addition to the value policy.")


def begin(category):
    global CATEGORY, CASES
    CATEGORY, CASES = category, []


def case(description, pattern, match, no_match, pitfall, flags="", contains=False):
    """Tests are authored explicitly; only whole-input boundary probes are automatic."""
    index = len(CASES) + 1
    if not contains:
        # JS $ also matches before a final newline. The end guard makes validation strict.
        pattern = "^" + pattern + "$(?![\\s\\S])"
        no_match = list(no_match) + ["!" + match[0], match[0] + "!", match[0] + "\n",
                                     match[0]+"\r", match[0]+"\r\n", match[0]+"\u2028", match[0]+"\u2029",
                                     " "+match[0], match[0]+" "]
    record = dict(
        id=f"gpt-{CATEGORY}-{index:02d}",
        difficulty="simple" if index <= 5 else "medium" if index <= 13 else "hard",
        status="ok", description=description,
        regex="/" + pattern + "/" + flags,
        match=list(dict.fromkeys(match)), no_match=list(dict.fromkeys(no_match)), pitfall=pitfall,
    )
    CASES.append(record)


def finish():
    assert len(CASES) == 20, (CATEGORY, len(CASES))
    # Upgrade the remaining Unicode whole-name cases to interacting cluster,
    # mark-attachment and separator rules rather than alphabet-only hard cases.
    cluster_specs = {
        "code-syntax": ("An international code-review label", "tag_", ""),
        "files-paths": ("An international basename", "", ".txt"),
        "logs": ("A localized log source", "src=", ";ok"),
        "measurements": ("A localized unit name", "unit:", ":0000"),
        "money": ("A currency display name", "", ""),
        "phone-numbers": ("A dial prefix name", "", ":0000"),
        "structured-data": ("An international mapping key", "{", ":ok}"),
        "multiline": ("A localized two-line label", "title=", "\nstatus=ok"),
    }
    if CATEGORY in cluster_specs:
        topic, prefix, suffix = cluster_specs[CATEGORY]
        unicode_name(topic, prefix, suffix)
        replacement = CASES.pop()
        replacement["id"] = CASES[15]["id"]
        replacement["difficulty"] = "hard"
        CASES[15] = replacement
    extra = {
        "dates-times": {14: ["due:aa:aa/aa:aa"], 18: ["time=aa:aa\ncopy=aa:aa"]},
        "money": {9: ["GBP 1.0"], 14: ["USD:1.0/1.0", "USD:1.zz/1.zz"]},
        "files-paths": {14: ["same:'ab.csv'/'ab.csv'", "same:'abcdefghi.csv'/'abcdefghi.csv'", "same:'ab!.csv'/'ab!.csv'"], 18: ["file='ab.txt'\ncopy='ab.txt'", "file='abcdefghi.txt'\ncopy='abcdefghi.txt'", "file='ab!.txt'\ncopy='ab!.txt'"]},
        "identifiers": {9: ["JOB/1000"], 14: ["same:00000:00000", "same:0000000:0000000"], 18: ["id=AAA\ncopy=AAA", "id=AAAAA\ncopy=AAAAA"]},
        "network": {10: ["tcp://node.test:90xx"], 15: ["192.168.4.1000/24"]},
        "web": {6: ["http://preview.local:80xx/"]},
        "phone-numbers": {6: ["x2000"], 20: ["from=10000;to=1001", "from=1000;to=10001"]},
        "postal-official": {19: ["AAA:01/AAA:02", "AA:01/AA:022"], 20: ["APP-123456", "APP-1234567890"]},
        "logs": {20: ["BEGIN ?abc?\nEND ?abc?", "BEGIN 'ab!'\nEND 'ab!'", "BEGIN 'abc'\nEND 'abc'\n"]},
        "multiline": {20: ["?abc?\n1234\n?abc?", "'ab!'\n1234\n'ab!'", "'abc'\n12a4\n'abc'"]},
        "unicode": {10: ["abcdefgh ijklmnopq"], 15: ["Ab!/Ab!"]},
        "exclusions": {18: ["key='root'\ncopy='root'", "key='null'\ncopy='null'"]},
        "passwords": {14: ["reset:'aa000000'/'aa000000'", "reset:'AA000000'/'AA000000'"]},
    }
    for index, tests in extra.get(CATEGORY, {}).items():
        c = CASES[index-1]
        c["no_match"] = list(dict.fromkeys(c["no_match"] + tests))
    # Vary the request register without changing any acceptance rule.
    for index, c in enumerate(CASES, 1):
        if index == 6:
            c["description"] = "Form-validation ticket: " + c["description"]
        elif index == 10:
            c["description"] = "Import gate — reject any nonconforming input. " + c["description"]
        elif index == 11:
            c["description"] = "Search filter, not whole-input validation. " + c["description"]
        elif index == 15:
            c["description"] = "Code review requirement: " + c["description"]
        elif index == 19:
            c["description"] = "Acceptance criteria:\n- " + c["description"] + "\n- Evaluate the input as supplied; do not trim or normalize it."
    lines = [f"# Benchmark GPT, catégorie {CATEGORY}. Format et contrôles : scripts/benchmark.py."]
    def yaml_string(value):
        # YAML does not combine JSON's UTF-16 surrogate escape pairs. Emit its
        # native eight-digit code-point escape for supplementary characters.
        encoded = json.dumps(value, ensure_ascii=False)
        return "".join(ch if ord(ch) < 128 else (f"\\u{ord(ch):04x}" if ord(ch) <= 0xffff else f"\\U{ord(ch):08x}") for ch in encoded)
    for c in CASES:
        lines += [f"- id: {c['id']}", f"  difficulty: {c['difficulty']}", "  status: ok", "  description: >-"]
        lines += ["    " + line for line in c['description'].splitlines()]
        lines += ["  regex: " + json.dumps(c['regex'], ensure_ascii=False)]
        for key in ("match", "no_match"):
            lines += [f"  {key}:"] + ["    - " + yaml_string(s) for s in c[key]]
        lines += ["  pitfall: >-"] + ["    " + line for line in c['pitfall'].splitlines()]
    (ROOT / f"{CATEGORY}.yaml").write_text("\n".join(lines) + "\n")
