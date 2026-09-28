# Tariffs and bill simulation

Tariffs are stored as [OpenEI URDB](https://openei.org/services/doc/rest/util_rates/?version=8)
JSON plus a few extension keys, parsed by `vpp.tariffs.urdb` into billable
components, and used in three places:

- the bill simulator (`POST /api/v1/tariffs/{id}/simulate`,
  `POST /api/v1/tariffs/simulate`),
- the customer-portal bill (`GET /api/v1/customer/me/bill`, and
  `GET /api/v1/customers/{id}/bill` for staff),
- price-aware optimization (`POST /api/v1/optimization/schedule` with a
  `tariff_id`).

**Maturity: beta.** The engine is unit-tested against hand-computed bills,
but it is not a certified billing system; see the known simplifications at
the end of this page.

## Managing tariffs

| Route | Who |
|---|---|
| `GET /api/v1/tariffs`, `GET /api/v1/tariffs/{id}` | any operator-side role |
| `POST`, `PUT`, `DELETE /api/v1/tariffs[/{id}]` | admin |
| `GET /api/v1/tariffs/presets[/{id}]` | any operator-side role |
| `GET /api/v1/tariffs/import-urdb` | reports whether `OPENEI_API_KEY` is configured |
| `POST /api/v1/tariffs/import-urdb` | admin; fetches a tariff by URDB label (needs `OPENEI_API_KEY`; OpenEI network errors map to `502`) |

Create and update reject URDB JSON the engine cannot bill (`422`), so a
stored tariff is always billable.

`TariffRead` includes derived, read-only fields computed from the same
parsed components the bill engine uses: `components` (name, kind, unit,
rate or tiers, sell rate, human-readable schedule), `tou_heatmap` and
`tou_heatmap_weekend` (12 months × 24 hours, $/kWh), `is_tou`, `sector`,
`source`, `description`, `nem_regime`/`nem_source`, and `parse_error`.

### Presets

The wheel ships four presets (`src/vpp/tariffs/presets/`): two real-world
tariffs captured at a stated date (PG&E E-TOU-C, SCE TOU-D-PRIME) and two
clearly labelled **illustrative** ones (residential tiered, commercial TOU
with demand charges). Presets are starting points; check rates against the
utility before relying on them.

## Supported URDB fields

| Field | Use |
|---|---|
| `energyratestructure` | periods × tiers of `{rate, max?, adj?, sell?}`; `rate + adj` is billed; `sell` is the export rate |
| `energyweekdayschedule`, `energyweekendschedule` | 12 × 24 period indices (month × local hour) |
| `demandratestructure`, `demandweekdayschedule`, `demandweekendschedule` | TOU demand charges |
| `flatdemandstructure`, `flatdemandmonths` | monthly (non-TOU) demand charges |
| `fixedchargefirstmeter`, `fixedchargeunits` | fixed charge, `$/month` or `$/day` |
| `mincharge` | minimum bill |
| `taxes` | list of `{name, rate, jurisdiction?, applies_to?}` |
| `utility`, `name`, `sector`, `startdate`, `source` | metadata |

A multi-period TOU tariff whose periods also have usage tiers bills each
period against its own cumulative kWh for the billing cycle (URDB's
per-period tier convention).

Not supported: demand lookback windows (`lookback*`, `demandwindow`),
non-USD currencies.

### Extension keys

These are not part of URDB; they live in the same JSON object.

| Key | Meaning |
|---|---|
| `adders` | list of `{name, rate, basis, ...}` surcharges (e.g. a percentage of the subtotal) |
| `nem` | the tariff's export-compensation regime: `none`, `nem2`, `nem3` or `net_billing` (spellings like `NEM-2`, `nem 3.0`, `NBT`, `net billing` are normalised) |
| `nem3_avoided_cost` | list of $/kWh export credits by **local hour of day** for `nem3` |

Example:

```json
{
  "name": "Residential TOU with NBT",
  "utility": "Example Utility",
  "sector": "Residential",
  "energyratestructure": [[{"rate": 0.32}], [{"rate": 0.48, "sell": 0.08}]],
  "energyweekdayschedule": [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0], "... 12 rows ..."],
  "energyweekendschedule": ["... 12 rows of 24 ..."],
  "fixedchargefirstmeter": 0.39,
  "fixedchargeunits": "$/day",
  "nem": "nem3",
  "nem3_avoided_cost": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.06, 0.07, 0.04, 0.03,
                        0.02, 0.02, 0.02, 0.02, 0.03, 0.05, 0.12, 0.25, 0.30, 0.22,
                        0.12, 0.08, 0.06, 0.05]
}
```

## Export credit (NEM)

`vpp.tariffs.nem` turns exported energy into a single credit line item per
billing cycle. It is shared by the simulator and the customer-portal bill,
so both credit exports identically.

| Regime | Credit |
|---|---|
| `none` | exports earn nothing |
| `nem2` | each exported kWh at the TOU period rate in effect when it was exported: the period's `sell` rate if the tariff defines one, else the retail import rate. Tariffs without TOU periods fall back to the bill's blended energy rate |
| `nem3` | each exported kWh at `nem3_avoided_cost[local hour]` |
| `net_billing` | only at explicit `sell` rates; exports in periods without a `sell` rate are not credited |

### Where the regime comes from

1. the request (`nem` on a simulation, `?nem=` on the staff bill endpoint —
   a what-if override), source `request`;
2. the tariff's `nem` extension key, source `tariff`;
3. URDB `dgrules`, source `urdb_dgrules`:

   | `dgrules` | regime |
   |---|---|
   | `Net Metering` | `nem2` |
   | `Net Billing Instantaneous`, `Net Billing Hourly`, `Buy All Sell All` | `net_billing`, or `nem3` when the tariff has `nem3_avoided_cost` |

4. otherwise `none` (source `default`).

`nem3` without an avoided-cost vector (from the request's
`nem3_avoided_cost` or the tariff's) is an error (`422`).

The avoided-cost vector is indexed by local hour of day
(`vector[hour % len(vector)]`): a 24-entry vector repeats daily. Entries
beyond the 24th are never used, so an 8760-hour vector is **not** applied
hour-of-year.

## Simulating a bill

`POST /api/v1/tariffs/{id}/simulate` bills a stored tariff;
`POST /api/v1/tariffs/simulate` takes either `tariff_id` or an inline
`urdb_json`. Any operator-side role may simulate. Exactly one load source is
required:

| Source | Fields | Billing window |
|---|---|---|
| `meter_trace` | `timestamps` (tz-aware interval starts), `import_kwh`, optional `export_kwh`, `interval_minutes` | `billing_period_start` / `billing_period_end` (required) |
| `synthetic` | `true`, or `{profile: residential\|commercial, avg_kw, pv_kw, interval_minutes: 15\|30\|60}`; `profile` defaults from the tariff sector | `period_days` (default 30) from `billing_period_start` (default: first day of the current month in `timezone`) |
| `csv` | the CSV text (up to 8 MB) | the span of the data, unless given |

Other fields: `timezone` (IANA, default `UTC`; used for TOU periods,
cycle boundaries and NEM3 hours), `billing_cycle` (`auto` \| `single` \|
`monthly`), `nem`, `nem3_avoided_cost`, `compare_to` (another stored tariff
id billed on the same load).

The synthetic load is a deterministic, illustrative shape (with optional
rooftop PV producing exports), not a forecast of any real customer.

The response has the total, flat `line_items`, per-cycle totals (`cycles`),
`nem_regime`/`nem_source`, `export_kwh`/`export_credit`, `notes`,
`load_summary`, and `comparison` (same shape) when `compare_to` was given.
The work runs off the event loop.

### CSV format

- A header row, then one row per interval, sorted or not (rows are sorted;
  duplicate timestamps are rejected).
- Timestamp column: `timestamp` (aliases `time`, `datetime`,
  `interval_start`, `start`) holding the interval **start** in ISO 8601.
  Timestamps without an offset are interpreted in the simulation
  `timezone`.
- Energy, either:
  - `import_kwh` (aliases `kwh`, `energy_kwh`) and optional `export_kwh` —
    energy per interval, or
  - `kw` (aliases `demand_kw`, `load_kw`, `power_kw`) and optional
    `export_kw` — average power over the interval; a negative `kw` is
    treated as export.
- The interval length is the most common gap between rows; it must be 1–60
  minutes and divide an hour. At most 110 000 rows (a year of 5-minute data).

```csv
timestamp,import_kwh,export_kwh
2026-06-01T00:00:00-07:00,0.42,0
2026-06-01T01:00:00-07:00,0.38,0
2026-06-01T12:00:00-07:00,0.05,1.9
```

### Billing-cycle rules

- `single`: one bill for the whole window.
- `monthly`: always split into cycles.
- `auto` (default): split only when the window is longer than 31 days.
- Cycles are anchored on the window's start day in the simulation timezone
  (like a meter-read cycle): a window starting on the 15th produces cycles
  15th → 15th. On months without that day, the last day of the month is
  used.
- Each cycle is billed separately (fixed and minimum charges once per
  cycle, demand charges on each cycle's own peak), the export credit is
  applied per cycle, and totals are summed. With several cycles, demand
  line items are merged with unit `kW-mo`.
- A trailing partial cycle is billed as a full cycle: fixed and minimum
  charges are **not** prorated. The per-cycle breakdown makes this visible.

## Customer-portal bill

`GET /api/v1/customer/me/bill` bills the customer's own stored revenue-meter
readings (`POST /api/v1/sites/{id}/meter-readings`) against the tariff
assigned to their customer profile, for a local calendar month, and
reports data coverage. The export regime comes from the assigned tariff as
above. When no tariff is assigned (or there is nothing to bill) the API
answers `409` with an explanation instead of inventing a bill.

## Known simplifications

- Credits are applied within a single billing cycle: no month-to-month
  roll-over and no annual true-up. A credit may take a bill below the
  tariff's minimum charge.
- Partial cycles are not prorated (see above).
- NEM3 avoided cost is hour-of-day only (see above).
- Demand lookback windows and non-USD currencies are not modelled.
