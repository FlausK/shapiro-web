from flask import Flask, render_template, request
from scipy.stats import shapiro
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            # 数字だけ抽出（カンマ・スペース区切りに対応）
            nums = [float(x) for x in raw_data.replace(',', ' ').split()]
            if len(nums) < 3:
                result = "データ数が少なすぎます。3つ以上入力してください。"
            else:
                stat, p = shapiro(nums)
                result = f"Shapiro-Wilk検定: p値 = {p:.4f} → {'正規分布' if p > 0.05 else '正規分布ではない'}"
        except ValueError:
            result = "数値の読み取りに失敗しました。カンマまたはスペース区切りで数値を入力してください。"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
