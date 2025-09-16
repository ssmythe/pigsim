# pigsim

Pig dice game simulator with parallel round-robin engine matchups and head-to-head matrix.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pigsim.py
```

## Results from 100,000 games each match

```text
Final Results (with 95% CI)
Rank  Player             Wins      Win %      ±CI
--------------------------------------------------
#1    LIN_GAP         2103433     52.59%    0.05%
#2    VAR_CAP         2098094     52.45%    0.05%
#3    PT              2093899     52.35%    0.05%
#4    H@25            2088032     52.20%    0.05%
#5    PP              2083704     52.09%    0.05%
#6    H@25P           2083701     52.09%    0.05%
#7    PH+PR           2075370     51.88%    0.05%
#8    ADP             2055718     51.39%    0.05%
#9    R3_then_20      2038324     50.96%    0.05%
#10   H@20            2036128     50.90%    0.05%
#11   SURP+10         2026449     50.66%    0.05%
#12   GAP             2014664     50.37%    0.05%
#13   P20_20_15       2010743     50.27%    0.05%
#14   R6              1990796     49.77%    0.05%
#15   RAND            1950422     48.76%    0.05%
#16   P25_20_15       1945754     48.64%    0.05%
#17   R5              1943552     48.59%    0.05%
#18   CEG25_15        1918270     47.96%    0.05%
#19   CEG             1908050     47.70%    0.05%
#20   H@15            1801186     45.03%    0.05%
#21   R4              1733711     43.34%    0.05%
```
