from flask import Flask, render_template, request, send_file
from scipy.stats import shapiro, skew, kurtosis, probplot, weibull_min
from scipy.special import gamma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import base64

from statsmodels.stats.power import TTestIndPower
from math import log, ceil
from scipy.stats import norm



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
last_data = []


@app.route('/')
def index():
    return render_template("index.html", title="Statistical Toolkit", heading="Welcome to the Statistical Toolkit")


# --- Shapiro-Wilk 正規性検定 ---
@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_page():
    global last_data
    table_html, message, show_plot = '', '', False
    decision_label, label_color = '', ''
    plot_data = ''

    if request.method == 'POST':
        raw_data = request.form.get('data', '')
        try:
            nums = np.array([float(x) for x in raw_data.replace(',', ' ').split()])
            last_data = nums
            w_stat, p = shapiro(nums)
            decision_label = 'Normally distributed ✅' if p > 0.05 else 'Not normally distributed ⚠'
            label_color = 'success' if p > 0.05 else 'danger'
            show_plot = True

            # 統計表
            df = pd.DataFrame({
                'Parameter': ['P-value', 'W', 'Size', 'Average', 'Median', 'Std Dev', 'Skew', 'Kurtosis'],
                'Value': [
                    round(p, 8), round(w_stat, 4), len(nums),
                    round(np.mean(nums), 4), round(np.median(nums), 4),
                    round(np.std(nums, ddof=1), 5), round(skew(nums), 3), round(kurtosis(nums), 3)
                ]
            })
            table_html = df.to_html(index=False, classes="result-table", border=0, escape=False)

            # 画像生成 → base64エンコードしてHTMLに埋め込み
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].hist(nums, bins='auto', color='skyblue', edgecolor='black')
            axs[0].set_title('Histogram')
            axs[1].set_title("QQ Plot")
            probplot(nums, dist="norm", plot=axs[1])
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            plot_data = base64.b64encode(buf.read()).decode('utf-8')

        except:
            message = "⚠ Invalid input format. Please enter numbers separated by commas or spaces."

    return render_template("shapiro.html",
                           title="Shapiro-Wilk Test",
                           heading=None,
                           table_html=table_html,
                           message=message,
                           show_plot=show_plot,
                           decision_label=decision_label,
                           label_color=label_color,
                           plot_data=plot_data)



# --- Weibull 寿命分析 ---
@app.route('/lifespan', methods=['GET', 'POST'])
def lifespan_page():
    result = ''
    lifespan_input = request.form.get('lifespan_data', '').strip()
    failure_cost = request.form.get('failure_cost', '')
    maint_cost = request.form.get('maint_cost', '')
    cost_plot_path = ''
    cdf_path = ''
    pdf_path = ''

    if request.method == 'POST' and lifespan_input:
        try:
            failures = np.array([float(x) for x in lifespan_input.replace(',', ' ').split()])
            shape, loc, scale = weibull_min.fit(failures, floc=0)
            beta, eta = round(shape, 3), round(scale, 3)
            avg_life = round(eta * gamma(1 + 1 / beta), 2)
            result = f"Weibull Estimation: β ≈ {beta}, η ≈ {eta}, Mean ≈ {avg_life}"

            # Preventive timing advice
            tip = f"Recommended inspection start: around {round(avg_life * 0.75, 2)} units"

            # Generate CDF and PDF graphs
            os.makedirs('static', exist_ok=True)
            t_vals = np.linspace(0, max(failures) * 1.5, 200)
            cdf = weibull_min.cdf(t_vals, beta, scale=eta)
            pdf = weibull_min.pdf(t_vals, beta, scale=eta)

            cdf_path = os.path.join('static', 'weibull_cdf.png')
            pdf_path = os.path.join('static', 'weibull_pdf.png')

            plt.figure(figsize=(6, 4))
            plt.plot(t_vals, cdf)
            plt.title("Weibull CDF")
            plt.xlabel("Time")
            plt.ylabel("Cumulative Probability")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(cdf_path)
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(t_vals, pdf)
            plt.title("Weibull PDF")
            plt.xlabel("Time")
            plt.ylabel("Density")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(pdf_path)
            plt.close()

            # Cost model only when both costs are provided
            if failure_cost and maint_cost:
                fc, mc = float(failure_cost), float(maint_cost)

                max_t = max(failures)
                t_vals = np.linspace(1, max_t, 100)
                failure_probs = weibull_min.cdf(t_vals, beta, scale=eta)
                exchange_freq = max_t / t_vals

                fail_costs = failure_probs * fc * exchange_freq
                maint_costs = mc * exchange_freq
                total_costs = fail_costs + maint_costs

                best_index = np.argmin(total_costs)
                best_t = t_vals[best_index]
                result += f"<br><b>Optimal maintenance interval:</b> {round(best_t, 2)} units"

                # Save cost curve
                cost_plot_path = os.path.join('static', 'cost_curve.png')
                plt.figure(figsize=(6, 4))
                plt.plot(t_vals, fail_costs, label='Failure Cost', color='deeppink')
                plt.plot(t_vals, maint_costs, label='Preventive Action Cost', color='dodgerblue')
                plt.plot(t_vals, total_costs, label='Total Impact', color='green')
                plt.axvline(best_t, color='gray', linestyle='--', label='Optimum')
                plt.xlabel("Maintenance Interval (t)")
                plt.ylabel("Cost (per year)")
                plt.title("Business Impact vs Maintenance Frequency")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(cost_plot_path)
                plt.close()

        except ValueError:
            result = "\u26a0 Invalid data format. Please enter numerical values separated by commas or spaces."
        except Exception as e:
            result = f"\u26a0 Unexpected error: {e}"

    return render_template("lifespan.html", title="Lifespan Analysis", heading=None,
                           lifespan_input=lifespan_input, failure_cost=failure_cost,
                           maint_cost=maint_cost, result=result,
                           tip=tip if 'tip' in locals() else '',
                           cdf_path=cdf_path, pdf_path=pdf_path,
                           cost_plot_path=cost_plot_path)


# --- サンプルサイズ計算 ---
@app.route('/samplesize', methods=['GET', 'POST'])
def sample_size_page():
    result = ''
    if request.method == 'POST':
        purpose = request.form.get('purpose')
        alpha = float(request.form.get('alpha', 0.05))

        if purpose == 'cpk':
            cpk_target = float(request.form.get('cpk_target', 1.33))
            z = norm.ppf(1 - alpha / 2)
            result = ceil((z / (cpk_target / 3))**2)

        elif purpose == 'ttest':
            effect_size = float(request.form.get('effect_size', 0.5))
            power = float(request.form.get('power', 0.8))
            analysis = TTestIndPower()
            result = ceil(analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided'))

        elif purpose == 'reliability':
            confidence = float(request.form.get('confidence', 0.95))
            failure_rate = float(request.form.get('failure_rate', 0.1))
            result = ceil(log(1 - confidence) / log(1 - (1 - failure_rate)))

        elif purpose == 'proportion':
            p = float(request.form.get('expected_prop', 0.95))
            margin = float(request.form.get('margin', 0.05))
            z = norm.ppf(1 - alpha / 2)
            result = ceil((z**2 * p * (1 - p)) / (margin**2))

    return render_template("samplesize.html", title="Sample Size Calculator", result=result)



if __name__ == '__main__':
    app.run(debug=True)
