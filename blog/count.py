# Read your markdown file
with open("sac.qmd", "r", encoding="utf-8") as f:
    text = f.read()

# Remove markdown syntax roughly
import re
clean_text = re.sub(r'[#\*\[\]\(\)!>`]', '', text)

# Count words
word_count = len(clean_text.split())
print(word_count)

import re
# Remove code chunks
text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)

# Remove image links
text_clean = re.sub(r'!\[.*?\]\(.*?\)', '', text_no_code)

# Remove other markdown syntax roughly
text_clean = re.sub(r'[#\*\[\]>`]', '', text_clean)

# Count words
word_count = len(text_clean.split())
print(word_count)