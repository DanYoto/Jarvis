import re
from typing import List, Optional
import cn2an


def normalize_numbers_chinese(text: str) -> str:
    """
    Normalize numbers in Chinese text to a consistent format.
    """

    def convert_number(match):
        number = match.group()
        try:
            # process decimal numbers
            if "." in number:
                integer_part, decimal_part = number.split(".")
                integer_chinese = cn2an.an2cn(int(integer_part))
                decimal_chinese = "点" + "".join(
                    [cn2an.an2cn(int(d)) for d in decimal_part]
                )
                return integer_chinese + decimal_chinese

            # process integer numbers
            num = int(number)
            if num == 0:
                return "零"
            elif 1000 <= num <= 9999:
                # For years, use the digit-by-digit reading method
                if re.search(
                    r"(19|20)\d{2}年",
                    match.string[max(0, match.start() - 2) : match.end() + 1],
                ):
                    return "".join([cn2an.an2cn(int(d)) for d in number])
                else:
                    return cn2an.an2cn(num)
            else:
                return cn2an.an2cn(num)

        except:
            # If conversion fails, return digit-by-digit reading method
            return "".join([cn2an.an2cn(int(d)) if d.isdigit() else d for d in number])

    # Match numbers (including decimals)
    text = re.sub(r"\d+\.?\d*", convert_number, text)
    return text


def normalize_special_symbols(text: str) -> str:
    unit_symbol_map = {
        "km/h": "公里每小时",
        "m/s": "米每秒",
        "km²": "平方公里",
        "m²": "平方米",
        "km": "公里",
        "cm": "厘米",
        "mm": "毫米",
        "ml": "毫升",
        "kg": "千克",
        "ha": "公顷",
        "℃": "摄氏度",
        "℉": "华氏度",
        "°": "度",
        "m": "米",
        "g": "克",
        "l": "升",
        "t": "吨",
    }

    general_symbol_map = {
        "%": "百分之",
        "‰": "千分之",
        "&": "和",
        "@": "艾特",
        "#": "井号",
        "$": "美元",
        "¥": "人民币",
        "€": "欧元",
        "£": "英镑",
        "+": "加",
        "-": "减",
        "×": "乘以",
        "÷": "除以",
        "=": "等于",
        "<": "小于",
        ">": "大于",
        "≤": "小于等于",
        "≥": "大于等于",
    }

    # match longest symbols first to avoid partial replacements
    sorted_unit_symbols = sorted(
        unit_symbol_map.items(), key=lambda x: len(x[0]), reverse=True
    )

    for symbol, replacement in sorted_unit_symbols:
        pattern = r"(\d+(?:\.\d+)?)\s*(" + re.escape(symbol) + r")(?![a-zA-Z0-9])"
        text = re.sub(pattern, r"\1" + replacement, text)

    text = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"百分之\1", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*‰", r"千分之\1", text)

    for symbol, replacement in general_symbol_map.items():
        if symbol not in ["%", "‰"]:
            text = text.replace(symbol, replacement)

    return text


def preprocess_text(text: str) -> str:
    """
    Process the input text to normalize numbers and special symbols.
    """
    # Normalize numbers
    text = normalize_numbers_chinese(text)

    # Normalize special symbols
    text = normalize_special_symbols(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_by_priority_endings(text: str) -> List[str]:
    """
    Split text into sentences based on priority endings.
    """
    # Define priority endings
    priority_endings = r"[。！？；;]"

    parts = re.split(f"({priority_endings})", text)
    sentences = []

    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            # Combine the part with the ending
            sentence = parts[i] + parts[i + 1]
            if sentence:
                sentences.append(sentence.strip())
        elif parts[i]:
            sentences.append(parts[i].strip())

    if len(sentences) > 1:
        return sentences

    return [text] if text.strip() else []


def force_split_by_length(text: str, max_length: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def split_by_secondary_endings(text: str, max_length: int = 30) -> List[str]:
    """
    Split text into sentences based on secondary endings.
    Contain as many sub-sentences as possible without exceeding max_length.
    """
    secondary_endings = [
        r"[，,](?=\s*[而且但是不过然而因此所以于是接着然后])",  # conjunctions after comma
        r"[，,](?=\s*[的地得])",  # common phrases after comma
        r"[，,]",  # ordinary comma
        r"[、]",  # enumeration comma
    ]
    for pattern in secondary_endings:
        parts = re.split(f"({pattern})", text)
        if len(parts) > 2:
            sub_sentences = []
            current_sentence = ""
            for i in range(0, len(parts), 2):
                if i + 1 < len(parts):
                    part = parts[i] + parts[i + 1]
                else:
                    part = parts[i]

                if not part.strip():
                    continue

                if len(current_sentence + part) <= max_length:
                    current_sentence += part
                else:
                    sub_sentences.append(current_sentence.strip())
                    current_sentence = part

            if current_sentence:
                sub_sentences.append(current_sentence.strip())

            if len(sub_sentences) > 1:
                return sub_sentences

    return force_split_by_length(text, max_length)


def split_text_intelligently(text: str, max_length: int = 256) -> List[str]:
    """
    Split text into chunks intelligently, grouping every two sentences together.
    Uses two-level priority-based sentence splitting.
    """
    text = preprocess_text(text)

    # primary split: by priority endings
    primary_sentences = split_by_priority_endings(text)

    print(f"Found {len(primary_sentences)} primary sentences")

    # Check if secondary splitting is needed
    processed_sentences = []
    for sentence in primary_sentences:
        if len(sentence) <= max_length:
            processed_sentences.append(sentence)
        else:
            sub_sentences = split_by_secondary_endings(sentence, max_length)
            processed_sentences.extend(sub_sentences)

    # Combine two sentences into one chunk in case sentence length is short
    chunks = []
    for i in range(0, len(processed_sentences), 2):
        if i + 1 < len(processed_sentences):
            two_sentences = processed_sentences[i] + processed_sentences[i + 1]
        else:
            two_sentences = processed_sentences[i]

        if len(two_sentences) <= max_length:
            chunks.append(two_sentences.strip())
        else:
            chunks.append(processed_sentences[i].strip())
            if i + 1 < len(processed_sentences):
                chunks.append(processed_sentences[i + 1].strip())

    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1} ({len(chunk)} chars): {chunk[:50]}...")

    return chunks


if __name__ == "__main__":
    sample_text = "春天的阳光透过窗棂洒在桌案上，微风轻拂过院中的桃花树，花瓣纷纷扬扬地飘落在青石板上。远处传来鸟儿清脆的啁啾声，仿佛在诉说着季节更替的喜悦。老人坐在藤椅上，手中捧着一杯热茶，茶香袅袅升起，与花香交融在一起，营造出一片宁静祥和的氛围。时光在这一刻仿佛放慢了脚步，让人不禁沉醉在这份简单而美好的生活中。"
    print("Original Text:", sample_text)
    chunks = split_text_intelligently(sample_text, max_length=30)
    print("Chunks:")
    for chunk in chunks:
        print(chunk)
