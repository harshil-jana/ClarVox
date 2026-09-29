import re

path = "/tmp/sih-anc-code/docx_unpacked/word/document.xml"
content = open(path, encoding="utf-8").read()

def heading2(text):
    return (f'<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:spacing w:after="120" w:before="300"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="0070C0"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
            f'<w:t xml:space="preserve">{text}</w:t></w:r></w:p>')

def qa(q, a):
    return (f'<w:p><w:pPr><w:spacing w:after="60" w:before="200"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1F497D"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
            f'<w:t xml:space="preserve">Q: {q}</w:t></w:r></w:p>'
            f'<w:p><w:pPr><w:spacing w:after="160" w:line="276"/><w:ind w:left="215"/></w:pPr>'
            f'<w:r><w:rPr><w:b/><w:bCs/><w:color w:val="0070C0"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
            f'<w:t xml:space="preserve">A: </w:t></w:r>'
            f'<w:r><w:rPr><w:color w:val="444444"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
            f'<w:t xml:space="preserve">{a}</w:t></w:r></w:p>')

def team_fill_in(text):
    return ('<w:tbl><w:tblPr><w:tblW w:type="pct" w:w="100%"/><w:tblBorders>'
            '<w:top w:val="single" w:color="auto" w:sz="4"/><w:left w:val="single" w:color="auto" w:sz="4"/>'
            '<w:bottom w:val="single" w:color="auto" w:sz="4"/><w:right w:val="single" w:color="auto" w:sz="4"/>'
            '<w:insideH w:val="single" w:color="auto" w:sz="4"/><w:insideV w:val="single" w:color="auto" w:sz="4"/>'
            '</w:tblBorders></w:tblPr><w:tblGrid><w:gridCol w:w="9000"/></w:tblGrid><w:tr><w:tc>'
            '<w:tcPr><w:tcW w:type="dxa" w:w="9000"/><w:shd w:fill="EAF1FB" w:color="auto" w:val="clear"/>'
            '<w:tcMar><w:top w:type="dxa" w:w="120"/><w:left w:type="dxa" w:w="160"/>'
            '<w:bottom w:type="dxa" w:w="120"/><w:right w:type="dxa" w:w="160"/></w:tcMar></w:tcPr>'
            '<w:p><w:r><w:rPr><w:b/><w:bCs/><w:color w:val="1F497D"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
            '<w:t xml:space="preserve">TEAM TO FILL IN: </w:t></w:r>'
            '<w:r><w:rPr><w:i/><w:iCs/><w:color w:val="444444"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
            f'<w:t xml:space="preserve">{text}</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
            '<w:p><w:pPr><w:spacing w:after="200"/></w:pPr></w:p>')

qas = [
    ("Today's demo has a wired USB mic and wired earphones with the speaker and listener standing right next to each other &#8211; how does this become usable at real combat distances where two people can't be physically wired together?",
     "We are not building a replacement for wireless communication &#8211; soldiers already carry radios or tactical headsets for the actual over-distance link, and re-engineering that is not the problem this project targets. Our system is a small AI processing stage designed to sit inline with that existing signal chain: on the transmitting side, it cleans the speaker's voice before it enters the radio; on the receiving side, it can equally clean up what comes out of the radio before it reaches the listener. Today's prototype is deliberately wired and co-located because that isolates the one variable we actually needed to prove &#8211; that the AI can denoise gunfire-grade noise in real time on lightweight embedded hardware. Radios already solve the distance/wireless problem; we did not need to re-solve it to demonstrate our contribution."),
    ("Have you tested this integrated with an actual radio or wireless communications device?",
     "Not yet &#8211; that is the next integration step, and it is a hardware-availability question (getting access to a tactical radio or comparable equipment to test with) rather than an open algorithm question. The processing itself is already proven to run well within the real-time budget on embedded hardware; wiring it into a specific radio's audio path is systems integration, not unsolved research."),
    ("How much additional delay would your processing add on top of whatever delay the radio link itself introduces?",
     "The frame hop is 16 ms, and measured model inference adds only 0.27 ms on top of that (about 55-60x faster than the real-time budget) &#8211; so our contribution to end-to-end delay is on the order of tens of milliseconds at most, small relative to typical radio or network latency, and not perceptible as lag in normal conversation."),
    ("Today's unit only has one microphone and one earphone &#8211; doesn't that mean you've only solved noise cleanup for one side of a conversation, not both directions?",
     "That's a fair and precise reading of today's demo: it shows one direction &#8211; cleanup at the point where a voice is captured, before it would be transmitted. The model itself, though, doesn't know or care where its noisy input comes from; it takes in noisy audio and outputs cleaned audio regardless of source. A fielded two-way system would place a second, identical, unmodified copy of this same pipeline on the receiving side, taking its input from the radio's received audio instead of a live microphone. That's the same code and the same trained model &#8211; there is no new engineering question there, just another instance of what is already demonstrated working."),
    ("If it generalizes that easily, why not just build and demo the two-way version now?",
     "Given the time available before presenting, we made a deliberate scope decision: prove the hard, novel part &#8211; real-time AI noise suppression on embedded hardware &#8211; as solidly as possible, rather than spend that time re-implementing wireless audio transport, which is a solved problem already handled by existing radio hardware and not something this project needs to reinvent to be valuable."),
    ("Would this replace a soldier's earpiece or radio?",
     "No &#8211; it's designed to supplement existing gear as an inline processing module, not replace the radio or headset a soldier already carries."),
]

section_xml = heading2("7.4 Wireless Communication, Range &amp; Two-Way Operation")
for q, a in qas:
    section_xml += qa(q, a)
section_xml += team_fill_in("an actual test with a real tactical radio or comparable two-way system, and a timeline for that integration, since this depends on hardware access your team may or may not have secured yet.")

marker = "<w:p><w:pPr><w:pStyle w:val=\"Heading1\"/><w:pBdr><w:bottom w:val=\"single\" w:color=\"0070C0\" w:sz=\"6\" w:space=\"4\"/></w:pBdr><w:spacing w:after=\"200\" w:before=\"400\"/></w:pPr><w:r><w:rPr><w:b/><w:bCs/><w:color w:val=\"1F497D\"/><w:sz w:val=\"30\"/><w:szCs w:val=\"30\"/></w:rPr><w:t xml:space=\"preserve\">8. How to Use This Document</w:t></w:r></w:p>"

assert marker in content, "marker not found"
assert content.count(marker) == 1, "marker not unique"

new_content = content.replace(marker, section_xml + marker)
open(path, "w", encoding="utf-8").write(new_content)
print("Inserted. New length:", len(new_content), "old length:", len(content))
