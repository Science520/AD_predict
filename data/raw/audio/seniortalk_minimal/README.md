---
license: cc-by-nc-sa-4.0
task_categories:
- automatic-speech-recognition
language:
- zh
pretty_name: SeniorTalk
size_categories:
- 10K<n<100K

features:
    - name: audio
      dtype: audio
    - name: transcription
      dtype: string
    - name: speaker
      dtype: string
    - name: gender
      dtype: string
    - name: age
      dtype: string
    - name: location
      dtype: string
    - name: device
      dtype: string

extra_gated_prompt: >-
  This dataset is made available for academic and non-commercial research purposes only. By accessing or using the dataset, you agree to comply with the following terms and conditions:  
  
  1. The dataset may only be used for academic research and educational purposes. Any commercial use, including but not limited to commercial product development, commercial speech recognition services, or monetization of the dataset in any form, is strictly prohibited.  

  2. The dataset must not be used for any research or applications that may infringe upon the privacy rights of the recorded participants. Any attempt to re-identify participants or extract personally identifiable information from the dataset is strictly prohibited. Researchers must ensure that their use of the dataset aligns with ethical research practices and institutional review board (IRB) guidelines where applicable.  
  
  3. If a participant (or their legal guardian) requests the removal of their data from the dataset, all recipients of the dataset must comply by deleting the affected data from their records. Researchers must acknowledge that such withdrawal requests may occur and agree to take reasonable steps to ensure compliance.  
  
  4. The dataset, in whole or in part, may not be redistributed, resold, or shared with any third party. Each researcher or institution must independently request access to the dataset and agree to these terms.  
  
  5. Any published work (e.g., papers, reports, presentations) that uses this dataset must properly cite the dataset as specified in the accompanying documentation.  
  
  6. You may create derived works (e.g., processed versions of the dataset) for research purposes, but such derivatives must not be distributed beyond your research group without prior written permission from the dataset maintainers.  
  
  7. The dataset is provided "as is" without any warranties, express or implied. The dataset maintainers are not responsible for any direct or indirect consequences arising from the use of the dataset.  
  
  8. Failure to comply with these terms may result in revocation of dataset access. The dataset maintainers reserve the right to deny access to any individual or institution found in violation of these terms.  
  
  9. If the researcher is employed by a for-profit, commercial entity, the researcher's employer shall also be bound by these terms and conditions, and the researcher hereby represents that they are fully authorized to enter into this agreement on behalf of such employer.
  
  By requesting access to the dataset, you acknowledge that you have read, understood, and agreed to these terms.  


extra_gated_fields:
  Name: text
  Email: text
  Affiliation: text
  Position: text
  Your Supervisor/manager/director: text
  I agree to the Terms of Access: checkbox
  I agree to use this dataset for non-commercial use ONLY: checkbox
---

# SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/BAAI/SeniorTalk)
[![arXiv](https://img.shields.io/badge/arXiv-2409.18584-b31b1b.svg)](https://www.arxiv.org/pdf/2503.16578)
[![License: CC BY-NC-SA-4.0](https://img.shields.io/badge/License-CC%20BY--SA--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Github](https://img.shields.io/badge/Github-SeniorTalk-blue)](https://github.com/flageval-baai/SeniorTalk)

## Introduction

**SeniorTalk** is a comprehensive, open-source Mandarin Chinese speech dataset specifically designed for research on  elderly aged 75 to 85. This dataset addresses the critical lack of publicly available resources for this age group, enabling advancements in automatic speech recognition (ASR), speaker verification (SV), speaker dirazation (SD), speech editing and other related fields.  The dataset is released under a **CC BY-NC-SA 4.0 license**, meaning it is available for non-commercial use.

## Dataset Details

This dataset contains 55.53 hours of high-quality speech data collected from 202 elderly across 16 provinces in China. Key features of the dataset include:

*   **Age Range:**  75-85 years old (inclusive).  This is a crucial age range often overlooked in speech datasets.
*   **Speakers:** 202 unique elderly speakers.
*   **Geographic Diversity:** Speakers from 16 of China's 34 provincial-level administrative divisions, capturing a range of regional accents.
*   **Gender Balance:**  Approximately 7:13 representation of male and female speakers, largely attributed to the differing average ages of males and females among the elderly.
*   **Recording Conditions:**  Recordings were made in quiet environments using a variety of smartphones (both Android and iPhone devices) to ensure real-world applicability.
*   **Content:**  Natural, conversational speech during age-appropriate activities.  The content is unrestricted, promoting spontaneous and natural interactions.
*   **Audio Format:**  WAV files with a 16kHz sampling rate. 
*   **Transcriptions:**  Carefully crafted, character-level manual transcriptions.  
* **Annotations:** The dataset includes annotations for each utterance, and for the speakers level.
    *   **Session-level**:  `sentence_start_time`,`sentence_end_time`,`overlapped speech`
    *   **Utterance-level**:  `id`, `accent_level`, `text` (transcription).
    *   **Token-level**:   `special token`([SONANT],[MUSIC],[NOISE]....)
    *   **Speaker-level**: `speaker_id`, `age`, `gender`, `location` (province), `device`.
      

### Dataset Structure

## Dialogue Dataset


The dataset is split into two subsets:
| Split      | # Speakers | # Dialogues | Duration (hrs) | Avg. Dialogue Length (h) |
| :--------- | :--------: | :----------: | :------------: | :-----------------------: |
| `train`    |    182     |    91    |     49.83     |           0.54            |
| `test`     |     20     |    10     |      5.70      |           0.57            |
| **Total**  |  **202**   |  **101**  |   **55.53**   |       **0.55**           |



The dataset file structure is as follows.
```

dialogue_data/  
├── wav  
│   ├── train/*.tar   
│   └── test/*.tar   
└── transcript/*.txt
UTTERANCEINFO.txt  # annotation of topics and duration
SPKINFO.txt   # annotation of location , age , gender and device
```
Each WAV file has a corresponding TXT file with the same name, containing its annotations.

For more details, please refer to our paper [SeniorTalk](https://www.arxiv.org/abs/2503.16578).

## ASR Dataset


The dataset is split into three subsets:
| Split      | # Speakers | # Utterances | Duration (hrs) | Avg. Utterance Length (s) |
| :--------- | :--------: | :----------: | :------------: | :-----------------------: |
| `train`    |    162     |    47,269    |     29.95      |           2.28            |
| `validation` |     20     |    6,891     |      4.09      |           2.14          |
| `test`     |     20     |    5,869    |      3.77     |           2.31            |
| **Total**  |  **202**   |  **60,029**  |   **37.81**   |       **2.27**           |


The dataset file structure is as follows.
```
sentence_data/  
├── wav  
│   ├── train/*.tar
│   ├── dev/*.tar 
│   └── test/*.tar   
└── transcript/*.txt   
UTTERANCEINFO.txt  # annotation of topics and duration
SPKINFO.txt   # annotation of location , age , gender and device
```
Each WAV file has a corresponding TXT, containing its annotations.

For more details, please refer to our paper [SeniorTalk](https://www.arxiv.org/abs/2503.16578).


## Dataset Access Control
This dataset is available to researchers upon request for academic and non-commercial use. To request access, please follow these steps:


1.  **Request Access on Hugging Face:** Make sure you are logged into your Hugging Face account and click the "Request access to this dataset" button on this page.
2.  **Submit Application via Email:** Send an email to **`your-research-email@example.com`** with the following information:
    * **Subject:** Dataset Access Request: [Your Name/Institution]
    * **Body:**
        * Your Hugging Face Username.
        * Your full name, title, and academic/institutional affiliation.
        * A link to your professional profile (e.g., university page, Google Scholar, LinkedIn).
        * A brief description of your research project and how you intend to use the dataset.
We will review your application and grant access on Hugging Face upon approval. Please allow 3-5 business days for processing.


##  📚 Cite me
```
@misc{chen2025seniortalkchineseconversationdataset,
      title={SeniorTalk: A Chinese Conversation Dataset with Rich Annotations for Super-Aged Seniors}, 
      author={Yang Chen and Hui Wang and Shiyao Wang and Junyang Chen and Jiabei He and Jiaming Zhou and Xi Yang and Yequan Wang and Yonghua Lin and Yong Qin},
      year={2025},
      eprint={2503.16578},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.16578}, 
}
```