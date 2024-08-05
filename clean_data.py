import json
import re
from tqdm import tqdm
from pydub import AudioSegment

dev_manifest = f"asds/mozilla-foundation_copy/common_voice_16_1/hy-AM/validation/validation_mozilla-foundation_common_voice_16_1_manifest.json"
test_manifest = f"asds/mozilla-foundation_copy/common_voice_16_1/hy-AM/test/test_mozilla-foundation_common_voice_16_1_manifest.json"
train_manifest = f"asds/mozilla-foundation_copy/common_voice_16_1/hy-AM/train/train_mozilla-foundation_common_voice_16_1_manifest.json"

def compute_char_counts(manifest):
  char_counts = {}
  with open(manifest, 'r') as fn_in:
      for line in tqdm(fn_in, desc="Compute counts.."):
          line = line.replace("\n", "")
          data = json.loads(line)
          text = data["text"]
          for word in text.split():
              for char in word:
                  if char not in char_counts:
                      char_counts[char] = 1
                  else:
                      char_counts[char] += 1
  return char_counts

import re
import json
from tqdm import tqdm

def clear_data_set(manifest, char_rate_threshold=None, normalize_audio=True):
    chars_to_ignore_regex = "[\.\,,\?\:\-!;՞()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„()՝,,։«»֊`՚’՜՛ehxlgnõ´av․'\u200b]"
    
    numbers_regex = "\d+"
    english_words_regex = "[a-zA-Z]{2,}"
    
    manifest_clean = manifest + '.clean'
    war_count = 0
    skip_count = 0
    long_audio_count = 0 

    with open(manifest, 'r') as fn_in, open(manifest_clean, 'w', encoding='utf-8') as fn_out:
        for line in tqdm(fn_in, desc="Cleaning manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            text = data["text"]
            audio_path = data['audio_filepath']


            if char_rate_threshold and len(text.replace(' ', '')) / float(data['duration']) > char_rate_threshold:
                print(f"[WARNING]: {audio_path} has char rate > 15 per sec: {len(text)} chars, {data['duration']} duration")
                war_count += 1
                continue
            
            if re.search(numbers_regex, text) or data['duration'] > 20:
                long_audio_count += 1
                continue
            
            if re.search(english_words_regex, text):
                skip_count += 1
                continue

            text = text.replace("եւ", "և")
            text = re.sub(chars_to_ignore_regex, "", text).lower()
            data["text"] = text

            if normalize_audio:
                audio = AudioSegment.from_file(audio_path)
                change_in_dBFS = -20.0 - audio.dBFS
                normalized_audio = audio.apply_gain(change_in_dBFS)
                normalized_audio.export(audio_path, format="wav")

            data = json.dumps(data, ensure_ascii=False)
            fn_out.write(f"{data}\n")

    print(f"[INFO]: {war_count} files were removed from manifest due to character rate, {long_audio_count} files were removed for being longer than 20 seconds, and an additional {skip_count} files were removed for containing numbers or English words.")

clear_data_set(dev_manifest)
clear_data_set(test_manifest)
clear_data_set(train_manifest, char_rate_threshold=15)