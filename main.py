import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def load_data():
    data_path = r"C:\Users\Bryce\Python Projects\Volatility"'\\'

    # old data is in xls format, new is in csv
    # default behaviour of parse_dates=True is to use the index (which we set to Date)
    current_vix_data = pd.read_csv(data_path + 'vixcurrent.csv', skiprows=1, parse_dates=True, index_col='Date')
    archived_vix_data = pd.read_excel(data_path + 'vixarchive.xls', skiprows=1, parse_dates=True, index_col='Date')
    current_vvix_data = pd.read_csv(data_path + 'vvixtimeseries.csv', skiprows=1, parse_dates=True, index_col='Date')

    # keep track of where data came from
    current_vix_data['is_archived'] = 0
    archived_vix_data['is_archived'] = 1
    # all data into one dataframe
    df = archived_vix_data.append(current_vix_data)
    df = pd.concat([df, current_vvix_data], axis=1, sort=False)
    # generate additional information
    df['high_low'] = df['VIX High'] - df['VIX Low']
    df['open_close'] = df['VIX Open'] - df['VIX Close']

    # generate grouping variables
    the_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df['day'] = pd.Categorical(df.index.day_name(), the_order)
    the_order = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
    df['month'] = pd.Categorical(df.index.month_name(), the_order)
    df['week'] = df.index.strftime("%V")

    df.rename(columns={'VVIX': 'VVIX Close'}, inplace=True)
    return df


def plot_timeseries(df):
    x = df.index
    y1 = df['high_low']
    plt.figure(0)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(x, y1)
    plt.title('High-Low')
    plt.show()

    x1 = df[df['is_archived'] == 0].index
    x2 = df[df['is_archived'] == 1].index
    y1 = df[df['is_archived'] == 0]['VIX Close']
    y2 = df[df['is_archived'] == 1]['VIX Close']
    plt.figure(1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.title('Vix Close')
    plt.show()

    plt.figure(2)
    y = df['VIX Close']
    y1 = df['VIX Close'].rolling(window=5).apply(lambda x: np.nanmean(x))
    # y2 = df['VIX Close'].rolling(window=30, min_periods=10).apply(lambda x: np.nanmean(x))
    # y3 = df['VIX Close'].rolling(window=100, min_periods=10).apply(lambda x: np.nanmean(x))
    plt.plot(x, y, label='VIX', alpha=0.3)
    plt.plot(x, y1, label='5MA', ls='dotted')
    # plt.plot(x, y2, label='30MA')
    # plt.plot(x, y3, label='100MA')
    plt.legend()
    plt.show()

    plt.figure(3)
    y = df['VIX Close']
    y1 = df['VIX Close'].ewm(span=5).mean()
    # y2 = df['VIX Close'].ewm(span=30).mean()
    # y3 = df['VIX Close'].ewm(span=100).mean()
    plt.plot(x, y, label='VIX', alpha=0.3)
    plt.plot(x, y1, label='5EMA', ls='dotted')
    # plt.plot(x, y2, label='30EMA')
    # plt.plot(x, y3, label='100EMA')
    plt.legend()
    plt.show()

    plt.figure(4)
    y = df['VIX Close']
    y1 = df['VVIX Close']
    plt.plot(x, y, label='VIX')
    plt.plot(x, y1, label='VVIX')
    plt.legend()
    plt.show()

    plt.figure(5)
    plt.scatter(y, y1)
    plt.xlabel('VIX')
    plt.ylabel('VVIX')
    plt.legend()
    plt.show()


def grp_plot(df, type='box', intraday=False):
    if intraday:
        col = 'high_low'
    else:
        col = 'VIX Close'

    # Group data
    day_grp = df.groupby('day')[col]
    week_grp = df.groupby('week')[col]
    month_grp = df.groupby('month')[col]

    if type == 'bar':
        day_grp = pd.DataFrame(day_grp.mean())
        week_grp = pd.DataFrame(week_grp.mean())
        month_grp = pd.DataFrame(month_grp.mean())

    plt.figure(0)
    if type == 'bar':
        day_grp.plot(kind=type, legend=None)
    else:
        temp = df.sort_values(by='day')
        temp.boxplot(column=col, by='day')
        plt.suptitle("")
        plt.xlabel("")

    if intraday:
        plt.title('Average VIX Width Per Day')
    else:
        plt.title('Average VIX Close Per Day')
    plt.xlabel('')
    plt.show()

    plt.figure(1)
    if type == 'bar':
        week_grp.plot(kind=type, legend=None)
    else:
        df.boxplot(column=col, by='week')
        plt.suptitle("")
        plt.xlabel("Week")

    if intraday:
        plt.title('Average VIX Width Per Week')
    else:
        plt.title('Average VIX Close Per Week')
    plt.show()

    plt.figure(2)
    if type == 'bar':
        month_grp.plot(kind=type, legend=None)
    else:
        temp = df.sort_values(by='month')
        temp.boxplot(column=col, by='month')
        plt.suptitle("")
    plt.xlabel("")

    if intraday:
        plt.title('Average VIX Width Per Month')
    else:
        plt.title('Average VIX Close Per Month')
    plt.show()


def build_dist(df):
    plt.figure(0)
    plt.hist(df['VIX Close'], bins='auto', density=True)
    plt.title('VIX Close')
    plt.show()

    bin_size = np.arange(start=0, stop=90, step=0.5)
    plt.figure(1)
    df.groupby('day')['VIX Close'].plot(kind='hist', bins=bin_size, alpha=.4, legend=True, density=True)
    plt.title('VIX Close by Day')
    plt.show()
    plt.figure(2)
    df.groupby('month')['VIX Close'].plot(kind='hist', bins=bin_size, alpha=.4, legend=True, density=True)
    plt.title('VIX Close by Month')
    plt.show()
    plt.figure(3)
    df.groupby('week')['VIX Close'].plot(kind='hist', bins=bin_size, alpha=.4, legend=True, density=True)
    plt.title('VIX Close by Week')
    plt.show()
    plt.figure(4)
    bin_size = np.arange(start=0, stop=20, step=0.1)
    df.groupby('day')['high_low'].plot(kind='hist', bins=bin_size, alpha=.4, legend=True, density=True)
    plt.title('High-Low')
    plt.show()


def mean_reverting(df):
    mu = df['VIX Close'].mean()  # long run VIX mean
    sig = df['VIX Close'].std()  # standard deviation of VIX
    theta = 1 / 2  # mean reversion rate
    weiner_var = sig * sig / (2 * theta)  # variance of the Weiner process
    x_0 = df['VIX Close'][0]  # start the process at the first VIX calculation

    T = 31  # time in years
    n = 250 * T  # points per year
    m = 100  # number of simulations (paths)
    delta_t = T / n  # time mesh

    x = np.ones((m, n + 1))
    x[:, 0] = x_0

    z = np.random.normal(0, 1, (m, n))  # standard normal observations
    # We take the absolute value to reflect the process off the X axis if it goes negative.
    # The negativity is due to the discretization, and because the Feller condition is violated
    # See: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405 for better implementation
    for i in range(0, n):
        x[:, i + 1] = np.abs(x[:, i] + theta * (mu - x[:, i]) * delta_t + sig * np.sqrt(delta_t) * z[:, i])

    return x


def double_mean_reverting(df):
    mu_1 = df['VIX Close'].mean()  # long run VIX mean
    mu_2 = df['VVIX Close'].mean()  # long run VVIX mean
    sig = df['VVIX Close'].std()  # standard deviation of VIX
    theta_1 = 1 / 2  # mean reversion rate on VIX
    theta_2 = 1 / 2  # mean reversion rate on VVIX
    x_0 = df[df['VIX Close'].notna()]['VIX Close'][0]  # start the process at the first VIX calculation
    v_0 = df[df['VVIX Close'].notna()]['VVIX Close'][0]  # start the process at the first VVIX calculation

    T = 31  # time in years
    n = 250 * T  # points per year
    m = 100  # number of simulations (paths)
    delta_t = T / n  # time mesh

    x = np.ones((m, n + 1))
    v = np.ones((m, n + 1))
    x[:, 0] = x_0
    v[:, 0] = v_0

    # correlate the standard normal random variables using VIX,VVIX correlation
    corr = df[['VIX Close', 'VVIX Close']].corr().values[0, 1]
    temp = np.random.multivariate_normal([0, 0], [[1, corr], [corr, 1]], (m, n))
    z_1 = temp[:, :, 0]
    z_2 = temp[:, :, 1]

    # We take the absolute value to reflect the process off the X axis if it goes negative.
    # The negativity is due to the discretization, and because the Feller condition is violated
    # See: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=946405 for better implementation
    for i in range(0, n):
        x[:, i + 1] = np.abs(
            x[:, i] + theta_1 * (mu_1 - x[:, i]) * delta_t + np.sqrt(v[:, i]) * np.sqrt(delta_t) * z_1[:, i])
        v[:, i + 1] = (v[:, i] + theta_2 * (mu_2 - v[:, i]) * delta_t + sig * np.sqrt(delta_t) * z_2[:, i])

    return x, v


def plot_sims(df, sim_df, col_name='VIX Close', title_name='VIX vs Sim', num_plots=5):
    if col_name == 'VIX Close':
        label = "VIX"
    else:
        label = "VVIX"

    plt.figure(0)
    plt.plot(df.index, df[col_name], label=label)
    for i in range(0, num_plots):
        plt.plot(df.index, sim_df[i, 0:len(df[col_name])], label='Sim ' + str(i), alpha=0.5)
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.hist(df[col_name], bins='auto', density=True, label=label)
    for i in range(0, num_plots):
        plt.hist(sim_df[i, 0:len(df[col_name])], bins='auto', density=True, alpha=0.4, label='Sim' + str(i))
    plt.legend()
    plt.show()


def corr_plot(df, vix_sim, vvix_sim, num_plots=3):
    plt.figure(0)
    plt.scatter(df['VIX Close'], df['VVIX Close'], label='Raw')
    for i in range(0, num_plots):
        plt.scatter(vix_sim[i, :], vvix_sim[i, :], alpha=0.2, label='Sim' + str(i))
    plt.xlabel('VIX')
    plt.ylabel('VVIX')
    plt.legend()
    plt.show()


def compare_stats(df, sim_df, sim_name, num_sims=3):
    data = np.array([df.mean(), df.std(), skew(df, nan_policy='omit').mean(), kurtosis(df, nan_policy='omit')])
    data_labels = ['VIX']
    for i in range(0, num_sims):
        data = np.vstack((data,np.array([sim_df[i, :].mean(), sim_df[i, :].std(), skew(sim_df[i, :], nan_policy='omit'),
                     kurtosis(sim_df[i, :], nan_policy='omit')])))
        data_labels = np.concatenate((data_labels, [sim_name + ' ' + str(i)]))
    print(pd.DataFrame(data, data_labels, ['Mean', 'StDev', 'Skew', 'Kurtosis']))


def main():
    df = load_data()
    # plot_timeseries(df)
    # build_dist(df)
    # grp_plot(df, type='bar', intraday=False)
    # grp_plot(df, type='box', intraday=False)
    static_vix = mean_reverting(df)  # MR simulation
    dynamic_vix, static_vvix = double_mean_reverting(df)  # double MR simulation
    plot_sims(df, static_vix, num_plots=3)
    plot_sims(df, dynamic_vix, num_plots=3)
    plot_sims(df, static_vvix, col_name='VVIX Close', num_plots=3)
    compare_stats(df['VIX Close'], static_vix, 'Static VIX')
    compare_stats(df['VIX Close'], dynamic_vix, 'Dynamic VIX')
    compare_stats(df['VVIX Close'], static_vvix, 'Static VVIX')

    # corr_plot(df, dynamic_vix, static_vvix)


if __name__ == "__main__":
    main()
