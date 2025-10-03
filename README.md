

# 貝茲曲線擬合系統

資訊科科展專題：  
**以特徵點主導之圖像輪廓分段貝茲曲線擬合與節點簡化系統**

這是一個能自動進行圖像輪廓擬合的系統。  
使用者可以在網頁上繪製筆畫，或直接上傳圖片，系統會自動擬合並轉換成 **SVG 格式**，方便後續編輯與應用。  

感謝你的使用 (>'-'<) 這份我真的做了好久，累(；′⌒`)

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

![網站介面_單筆畫模式](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%96%AE%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E4%BB%8B%E9%9D%A2%E5%9C%96.png)

![單筆畫模式截圖一](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%96%AE%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E4%B8%80.png)

![單筆畫模式截圖二](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%96%AE%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E4%BA%8C.png)

---

### 多筆畫模式

* 取消勾選 **單筆畫模式** 後，即可切換至 **多筆畫模式**。
* 此時可直接拖曳圖片進入網站。
* 點選 **發送到 Python**，根據筆畫複雜度，系統將花費約 **3~10 秒** 完成擬合。
* 擬合完成後，可點選 **下載 SVG**，將結果儲存為 **SVG 檔案**。

![網站介面\_多筆畫模式](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%A4%9A%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E4%BB%8B%E9%9D%A2%E5%9C%96.png)

![多筆畫模式_筆劃圖一](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%A4%9A%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E7%AD%86%E7%95%AB%E5%9C%96%E4%B8%80.png)

![多筆畫模式_筆劃圖二](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%96%AE%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E4%BA%8C.png)

![多筆畫模式_圖片擬合圖一](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%A4%9A%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E5%83%8F%E4%B8%80.png)

![多筆畫模式_圖片擬合圖一 SVG](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%A4%9A%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E5%83%8F%E4%B8%80SVG.png)

![多筆畫模式_圖片擬合圖二](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%A4%9A%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E5%83%8F%E4%BA%8C.png)

![多筆畫模式_圖片擬合圖二 SVG](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/img/%E5%A4%9A%E7%AD%86%E7%95%AB%E6%A8%A1%E5%BC%8F_%E5%9C%96%E5%83%8F%E4%BA%8CSVG.png)

---

## ⚠️ 注意事項

* 多筆畫模式的圖片上傳，**請盡量選擇以線條構成的圖像**。
* 系統目前 **不支援色塊填充與多顏色筆畫**，未來版本將持續更新，敬請期待！

---

##  專題工作書文本

[工作書](https://github.com/James-JAJ/Bezier_Fitting_Tool/blob/main/%E4%BB%A5%E7%89%B9%E5%BE%B5%E9%BB%9E%E4%B8%BB%E5%B0%8E%E4%B9%8B%E5%9C%96%E5%83%8F%E8%BC%AA%E5%BB%93%E5%88%86%E6%AE%B5%E8%B2%9D%E8%8C%B2%E6%9B%B2%E7%B7%9A%E6%93%AC%E5%90%88%E8%88%87%E7%AF%80%E9%BB%9E%E7%B0%A1%E5%8C%96%E7%B3%BB%E7%B5%B1.docx)

---

##  特別感謝

* [yenfugod]()




