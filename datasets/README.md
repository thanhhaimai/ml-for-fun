# Datasets

## Names

From: <https://download.pytorch.org/tutorial/data.zip>

## Vietnamese - English translation

- From <https://github.com/stefan-it/nmt-en-vi>
- From <https://tatoeba.org>

I don't reproduce it here because I don't know the original license.

NOTE: I skimmed through the datasets, and tbh, it's definitely useable, but not of the highest quality. There are many mispellings, and the sentence structure is sometimes awkward (trying to stick too close to English).

## Shakespeare

From: <https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt>

## IOI

Popular names are from Social Security

<https://www.ssa.gov/oact/babynames/decades/names2000s.html>

Extracted using

```shell
awk '{print $2; print $4}' download_names.txt | tee datasets/ioi/popular_names.txt
```
