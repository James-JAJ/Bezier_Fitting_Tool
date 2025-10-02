

# 貝茲曲線擬合系統

資訊科科展專題：  
**以特徵點主導之圖像輪廓分段貝茲曲線擬合與節點簡化系統**

這是一個能自動進行圖像輪廓擬合的系統。  
使用者可以在網頁上繪製筆畫，或直接上傳圖片，系統會自動擬合並轉換成 **SVG 格式**，方便後續編輯與應用。  

感謝你的使用 (>'-'<) 這份我真的做了好久，累死我了(；′⌒`)

---

## 必要模組

請先確認 Python 環境中安裝以下模組：

```python
import numpy as np
import cv2
import sys
import time
import threading
import base64

from scipy.interpolate import make_interp_spline
from scipy.spatial import procrustes, KDTree, cKDTree

from flask import Flask, request, jsonify, send_from_directory, Response

from io import BytesIO
from PIL import Image 
import svgwrite
````

---

## 使用說明

### 單筆畫模式

* 預設為 **單筆畫模式**，僅能繪製一筆線條。
* 若要清除筆畫，請按下 **清除** 鍵。
* 畫完後，點選 **發送到 Python**，系統會花約 **1–2 秒** 進行擬合。
* 擬合完成後，會將 **紅色曲線** 疊加在原圖上，並以 **綠點** 標示貝茲曲線節點。

![網站介面\_單筆畫模式]()
![單筆畫模式截圖一]()
![單筆畫模式截圖二]()

---

### 多筆畫模式

* 取消勾選 **單筆畫模式** 後，即可切換至 **多筆畫模式**。
* 此時可直接拖曳圖片進入網站。
* 點選 **發送到 Python**，根據筆畫複雜度，系統將花費約 **3–10 秒** 完成擬合。

![網站介面\_多筆畫模式]()
![多筆畫模式截圖一]()
![多筆畫模式截圖二]()
![多筆畫模式截圖三]()

* 擬合完成後，可點選 **下載 SVG**，將結果儲存為 **SVG 檔案**。

---

## ⚠️ 注意事項

* 多筆畫模式的圖片上傳，**請盡量選擇以線條構成的圖像**。
* 系統目前 **不支援色塊填充與多顏色筆畫**，未來版本將持續更新，敬請期待！

---

##  專題工作書文本

[工作書]()

---

##  特別感謝

* [yenfugod]()



# 使用說明



## 專題工作書文本

[Awesome README](https://github.com/matiassingers/awesome-readme)


## 特別感謝

 - [yenfugod]()



