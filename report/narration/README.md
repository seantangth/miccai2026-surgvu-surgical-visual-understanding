# 3 分鐘簡報旁白（ElevenLabs 用）

`slide1.txt` ～ `slide4.txt` 對應 `../APC_TDC_presentation.pptx` 的四張投影片。
`full.txt` 是四段接起來的完整稿，想一次生成整段音軌時用。

| 檔案 | 字數 | 預估長度 |
|---|---|---|
| slide1.txt | 49 | 18–20 秒 |
| slide2.txt | 137 | 51–55 秒 |
| slide3.txt | 162 | 60–65 秒 |
| slide4.txt | 106 | 39–42 秒 |
| **合計** | **454** | **168–182 秒** |

官方上限是 3 分鐘（180 秒）。上表的慢速端剛好貼到上限，**生成後務必看實際長度**；
超過就把 ElevenLabs 的 speed 調到 1.05，或把第四張最後那句「One honest caveat …」整段拿掉（省 23 字）。

## 稿子已經為 TTS 處理過

數字與縮寫全部寫成唸法，不要改回符號，否則會被唸錯：

- `0.493` → `zero point four nine three`
- `AP75` → `A P seventy five`
- `mAP` → `mean A P`
- `APC_TDC` → `A P C T D C`
- `OCR` → `O C R`

## 建議設定

- **模型**：Eleven Multilingual v2（穩定、口條適合簡報）；想要更有起伏可試 v3。
- **Stability** 約 50、**Similarity** 約 75、**Speed** 1.0。
- **建議一張一段分開生成**，四個音檔各自對齊投影片切換點，比單一長音軌好剪。

## 生成後要聽的三個地方

1. `R T DETR` — 可能被唸成單字。若不理想，改寫成 `R T detector` 再生成該段。
2. `prograsp` — 器械名，可能重音怪異。
3. `A P C T D C` — 確認是逐字母唸，不是拼成一個字。

投影片的備忘稿與這些檔案內容一致，由 `../make_slides.py` 自動帶入。

## 生成完音檔之後

把四個音檔命名為 `slide1`、`slide2`、`slide3`、`slide4`（副檔名 mp3／wav／m4a 皆可）放在同一個資料夾，然後：

```bash
python ../make_video.py ../APC_TDC_presentation.pptx <音檔資料夾> ../APC_TDC_presentation.mp4
```

腳本會把投影片以 4K 渲染再降到 1080p（字比較銳利），每張配上對應音軌，
串成一支 H.264 + AAC 的 MP4，最後印出總長並檢查是否超過 3 分鐘上限。
以假音檔實測過：1920×1080、30 fps、48 kHz 立體聲，176.6 秒約 8 MB。
