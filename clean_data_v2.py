import json
from tqdm import tqdm

# dev_manifest = f"/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/mozilla-foundation_copy_fleurs_mcv/common_voice_17_0/hy-AM/dev/validation_mozilla-foundation_common_voice_17_0_manifest.json"
test_manifest = f"/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds/fleurs_hy/test/test_aminfest.json"
# train_manifest = f"grqaser_hobbit_test.json"

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

import json
import re
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import split_on_silence


def clear_data_set(manifest, char_rate_threshold=None, silence_thresh=-40, min_silence_len=500, normalize_audio=True):
    chars_to_ignore_regex = r"[։֊.,?:\-!;՞()«»…\[\]/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„()՝»,:՝«`՚’՜՛ehxlgnõ´av․\u200b]"    
    numbers_regex = "\d+"
    english_words_regex = "[a-zA-Z]{2,}"
    wester_armenain_count = 0

    # western_armenian = ['կը','մէջ','մը','մէջը']
    
    manifest_clean = manifest + '.clean'
    war_count = 0
    skip_count = 0
    long_audio_count = 0 

    with open(manifest, 'r') as fn_in, open(manifest_clean, 'w', encoding='utf-8') as fn_out:
        for line in tqdm(fn_in, desc="Cleaning manifest data"):
            line = line.replace("\n", "")
            data = json.loads(line)
            text = data["text"]

            # data['audio_filepath'] = data['audio_filepath'].replace('mozilla-foundation', 'mozilla-foundation_copy')
            # data['audio_filepath'] = data['audio_filepath'].replace('/workspace/nemo_capstone/asds', '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/asds')
            # print("#$###################",audio_path)
            # exit()
            # data['audio_filepath'] = data['audio_filepath'].replace('/test/audio/', '/asds/mozilla-foundation_copy_fleurs_mcv/common_voice_17_0/hy-AM/test/hy-AM_test_0/')
            # data['audio_filepath'] = data['audio_filepath'].replace('/train/audio/', '/asds/mozilla-foundation_copy_fleurs_mcv/common_voice_17_0/hy-AM/train/hy-AM_train_0/')
            # data['audio_filepath'] = data['audio_filepath'].replace('/dev/audio/', '/asds/mozilla-foundation_copy_fleurs_mcv/common_voice_17_0/hy-AM/dev/hy-AM_validation_0/')
            # data['audio_filepath'] = data['audio_filepath'].replace('/other/audio/', '/asds/mozilla-foundation_copy_fleurs_mcv/common_voice_17_0/hy-AM/other/hy-AM_other_0/')
            # data['audio_filepath'] = data['audio_filepath'].replace('/home/asds/ml_projects_mher/NeMo-speech-data-processor/', '/home/asds/ml_projects_mher/NeMo-SDP-ArmData/')
            if char_rate_threshold and len(text.replace(' ', '')) / float(data['duration']) > char_rate_threshold:
                print(f"[WARNING]: {data['audio_filepath']} has char rate > 15 per sec: {len(text)} chars, {data['duration']} duration")
                war_count += 1
                continue
            

            # for i in western_armenian:
            #     if i in text.split():
            #         wester_armenain_count+=1
            #         continue
                    
            if re.search(english_words_regex, text):
                skip_count += 1
                continue

            # if normalize_audio or silence_thresh:
            #     audio = AudioSegment.from_file(data['audio_filepath'])
            #     # Remove silence
            #     # audio_chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
            #     # audio = sum(audio_chunks, AudioSegment.silent(duration=0))  # Combine all non-silent chunks

            #     # Update duration
            #     data['duration'] = len(audio) / 1000.0

            #     if normalize_audio:
            #         change_in_dBFS = -20.0 - audio.dBFS
            #         audio = audio.apply_gain(change_in_dBFS)
                
            #     audio.export(data['audio_filepath'], format="wav")

            
            if re.search(numbers_regex, text) or data['duration'] > 20 or data['duration'] < 1.5:
                long_audio_count += 1
                continue

            
            # words_count = len(text.split())
            # if words_count <= 3:
            #     continue
            
            text = re.sub(chars_to_ignore_regex, "", text).lower()
            text = text.replace("և", "եւ")
            data["text"] = text


            data = json.dumps(data, ensure_ascii=False)
            fn_out.write(f"{data}\n")

    print(f"[INFO]: {war_count} files were removed from manifest due to character rate, {long_audio_count} files were removed for being longer than 20 seconds, and an additional {skip_count} files were removed for containing numbers or English words.{wester_armenain_count} removes for containe western armenian texts")


# clear_data_set(dev_manifest)
# clear_data_set(test_manifest)
clear_data_set(test_manifest, char_rate_threshold=20)