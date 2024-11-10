from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key ="oafkndgojnsojgna"

def generate_plots(N, mu, sigma2, S, beta0, beta1):
    session["N"] = N
    session["mu"] = mu
    session["sigma2"] = sigma2
    session["S"] = S
    # and a random dataset Y with normal additive error (mean mu, variance sigma^2).
    X = np.random.rand(N)
    error = np.random.normal(mu, np.sqrt(sigma2), N)
    Y = beta0 + beta1 * (X + mu + error) 

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    session["slope"] = slope
    session["intercept"] = intercept
    # Generate a scatter plot of (X, Y) with the fitted regression line
    plt.figure()
    plt.scatter(X, Y, color="blue", alpha=0.5)
    plt.plot(X, slope * X + intercept, color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Regression Line: y = {slope:.2f}x + {intercept:.2f}")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Step 2: Run S simulations and create histograms of slopes and intercepts

    # Initialize empty lists for slopes and intercepts
    slopes = []
    intercepts = []

    # Run a loop S times to generate datasets and calculate slopes and intercepts
    for _ in range(S):
        # Generate random X values with size N between 0 and 1
        X_sim = np.random.rand(N)
        error = np.random.normal(mu, np.sqrt(sigma2), N)
        Y_sim = beta0 + beta1 * (X_sim + mu + error) 

        # Fit a linear regression model to X_sim and Y_sim
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)

        # Append the slope and intercept of the model to slopes and intercepts lists
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    session["slopes"] = slopes
    session["intercepts"] = intercepts
    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S
    session["slope_ext"] = slope_more_extreme
    session["inter_ext"] = intercept_more_extreme
    
    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["N"] = N
        # Generate plots and results
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S, beta0, beta1)

        return render_template("index.html", plot1=plot1, plot2=plot2,
                               slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve data from session
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Use the slopes or intercepts from the simulations
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0
    # Calculate p-value based on test type
    if test_type == "≠":
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= np.abs(observed_stat - hypothesized_value))
        print(0)
    elif test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
        print(1)
    else:
        p_value = np.mean(simulated_stats <= observed_stat)
        print(2)

    fun_message = "That's incredibly rare!" if p_value <= 0.0001 else None

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.hist(simulated_stats, bins=20, color="teal", alpha=0.7, edgecolor="black")
    plt.axvline(observed_stat, color="red", linestyle="dashed", linewidth=0.5, label="Observed Statistic")
    plt.axvline(hypothesized_value, color="purple", linestyle="solid", linewidth=0.5, label='Hypothesized Statistic')
    plt.title("Histogram of Simulated Statistics")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        slope_extreme=session["slope_ext"], intercept_extreme=session["inter_ext"],
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        p_value=p_value,
        fun_message=fun_message,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    t_critical = np.abs(np.percentile(estimates, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))


    ci_lower, ci_upper = mean_estimate - t_critical[1] * std_estimate / np.sqrt(S), mean_estimate + t_critical[0] * std_estimate / np.sqrt(S)
    includes_true = ci_lower <= true_param <= ci_upper

    plot4_path = "static/plot4.png"
    plt.scatter(range(len(estimates)), estimates, color="gray", alpha=0.5)
    plt.axhline(mean_estimate, color="blue", label="Mean Estimate")
    plt.axhline(true_param, color="red", linestyle="dashed", label="True Parameter")
    plt.fill_between(range(len(estimates)), ci_lower, ci_upper, color="lightblue", alpha=0.3)
    plt.xlabel("Number of Estimates")
    plt.ylabel(f"Estimated {parameter}")
    plt.title(f"Estimates and {int(confidence_level*100)}% Confidence Interval")
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        slope_extreme = session["slope_ext"],
        intercept_extreme = session["inter_ext"],
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
        true_param = true_param
    )
    
if __name__ == "__main__":
    app.run(debug=True)