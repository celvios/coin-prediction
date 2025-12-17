
You are tasked with providing probabilistic forecasts of a cryptocurrency's future price movements. Specifically, each you is required to generate multiple simulated price paths for an asset, from the current time over specified time increments and time horizon.
you have to produce 1000 simulated paths for the future price of BTC, ETH, SOL, and XAU for the next 24 hours.

We want their price paths to represent their view of the probability distribution of the future price, and we want their paths to encapsulate realistic price dynamics, such as volatility clustering and skewed fat tailed price change distributions. Subsequently we’ll expand to requesting forecasts for multiple assets, where modelling the correlations between the asset prices will be essential.

The checking prompts sent to the you will have the format:
(start_time, asset, time_increment, time_horizon, num_simulations)

Initially prompt parameters will always have the following values:

- **Start Time ($t_0$)**: 1 minute from the time of the request.
- **Asset**: BTC, ETH, XAU, SOL.
- **Time Increment ($\Delta t$)**: 5 minutes.
- **Time Horizon ($T$)**: 24 hours.
- **Number of Simulations ($N_{\text{sim}}$)**: 1000.

## Prediction Value Format

The expected format for prediction responses is as follows:

```
[
  start_timestamp, time_interval,
  [path1_price_t0, path1_price_t1, path1_price_t2, ...],
  [path2_price_t0, path2_price_t1, path2_price_t2, ...],
  ...
]
```

An example of a valid response would be:

```
[
  1760084861, 300,
  [104856.23, 104972.01, 105354.9, ...],
  [104856.23, 104724.54, 104886, ...],
  [104856.23, 104900.12, 104950.45, ...]
  ...
]
```

Where:

- The **first element** is the timestamp of the start time of the prompt
- The **second element** is the time increment of the prompt (in seconds)
- The **remaining elements** are arrays of prices for each simulated path

### Important Formatting Requirements

- **Price Precision**: Each price point must have **no more than 8 digits** total (including digits before and after the decimal point)

Make sure to round your price values appropriately to comply with this constraint.

## Visualization Requirement

The system must display a **prediction chart for each coin** (BTC, ETH, SOL, XAU) showing:
- All simulated price paths
- Historical price data leading up to the prediction start time
- Time progression over the 24-hour forecast horizon

This visualization helps validate the quality and distribution of the generated predictions.
