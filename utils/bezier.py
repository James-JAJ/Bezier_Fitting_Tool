import cv2
import numpy as np
#bezier check
def bezier_curve_calculate(points, num_of_points=50):
    """
    ✅ 功能：
        根據一組四個三次貝茲曲線控制點，產生該曲線上均勻抽樣的點（離散化表示）

    Args:
        points (list of tuple): 四個貝茲曲線控制點，例如 [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        num_of_points (int): 要生成多少個曲線上的點（預設為 50）

    Returns:
        curve_points (list of tuple): 回傳離散點組成的貝茲曲線 [(x1,y1), (x2,y2), ..., (xn,yn)]

    ⚠️ 備註：
        - 所有控制點的座標應為整數，否則結果會因浮點轉換造成不必要的偏差
        - 最後回傳的每個點會被轉成 `int`，所以如果精度有要求（如 SVG），建議自行保留 float
        - num_of_points 數值越大，曲線越平滑，但計算量也會提高

    💡 補充：
        這是標準的三次貝茲公式：
            B(t) = (1−t)^3 * P0 + 3(1−t)^2 * t * P1 + 3(1−t) * t^2 * P2 + t^3 * P3
        其中 t ∈ [0, 1]，以等距方式插值。
    """
    curve_points = []
    for t in np.linspace(0, 1, num_of_points):
        x = float((1 - t)**3 * points[0][0] + 3 * (1 - t)**2 * t * points[1][0] + 3 * (1 - t) * t**2 * points[2][0] + t**3 * points[3][0])
        y = float((1 - t)**3 * points[0][1] + 3 * (1 - t)**2 * t * points[1][1] + 3 * (1 - t) * t**2 * points[2][1] + t**3 * points[3][1])
        curve_points.append((int(x), int(y)))
    return curve_points

def draw_curve_on_image(image, curve_points, thickness=1, color=(0, 0, 255)):
    """
    ✅ 功能：
        將貝茲曲線的點集合（離散點）畫在一張影像上

    Args:
        image (ndarray): 要畫線的圖片（必須是 OpenCV 圖像格式，通常為 BGR）
        curve_points (list of tuple): 曲線的點座標 [(x1, y1), (x2, y2), ...]
        thickness (int): 線條粗細（預設為 1）
        color (tuple): 線條顏色，預設紅色 (0, 0, 255)

    Returns:
        image (ndarray): 回傳已經畫好線的圖片

    ⚠️ 備註：
        - `isClosed=False`，表示這條貝茲曲線不會自動閉合，如需封閉請自行連接首尾
        - 請確保 `curve_points` 中至少有兩個點，否則不會畫任何東西
        - 畫在原圖上，會改變傳入的 `image`；如需保留原圖，請先複製

    💡 補充：
        `cv2.polylines` 是用直線段連接點陣，並非真正的連續曲線。
        若想導出精確 SVG 貝茲，建議用 path 語法 `M...C...` 表達。
    """
    if len(curve_points) >= 2:
        pts = np.array(curve_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)
    return image

