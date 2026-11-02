"""
STEP 2 : Creating a model serving front end using Streamlit

This script should sit in the same folder as that of train_house_price_model.py and house_price_pipeline.pkl.

Here are the broad steps the code below performs:
  1. loads the house_price_pipeline.pkl and fetches pipeline and RMSE from it
  2. Projects a streamlit front end that collects multiple inputs from a human user
  3. These inputs are stored in a single row of a pandas DataFrame
  4. The row is sent to pipeline.predict() to get a price estimate
  5. The price estimate is converted to a price range using the RMSE
  6. The range is displayed to the user in a nice format.


Command to activate the streamlit front end : streamlit run house_price_app.py

"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")

CURRENCY = "$"          # change to "₹" etc. if your housing_data.csv uses another currency


# ----------------------------------------------------------------------
#  Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
    padding: 1.8rem 2rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.hero p {margin: .35rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.result-card {
    background: linear-gradient(120deg, #059669 0%, #10b981 100%);
    border-radius: 18px; padding: 1.6rem 2rem; color: white; margin: 1rem 0 .8rem 0;
    box-shadow: 0 8px 24px rgba(16,185,129,.28);
}
.result-label {font-size: .85rem; letter-spacing: .12em; text-transform: uppercase; opacity: .9;}
.result-range {font-size: 2.5rem; font-weight: 800; line-height: 1.15; margin: .15rem 0;}
.result-sub {font-size: .98rem; opacity: .95;}
.bar-wrap {margin: .4rem 0 1.2rem 0;}
.bar {position: relative; height: 12px; border-radius: 99px;
      background: linear-gradient(90deg, #fbbf24, #10b981, #fbbf24);}
.bar-marker {position: absolute; left: 50%; top: -5px; width: 22px; height: 22px; margin-left: -11px;
             border-radius: 50%; background: white; border: 4px solid #059669;
             box-shadow: 0 2px 8px rgba(0,0,0,.25);}
.bar-labels {display: flex; justify-content: space-between; font-size: .85rem;
             opacity: .75; margin-top: .45rem;}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,.10); border-radius: 14px; padding: .9rem 1.1rem;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
#  1. Load the pipeline + RMSE
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("house_price_pipeline.pkl")

artifact = load_artifact()
pipeline = artifact["pipeline"]
rmse = artifact["rmse"]


# ----------------------------------------------------------------------
#  Read the allowed values straight from the trained pipeline, so the
#  dropdowns always match what the model was trained on.
# ----------------------------------------------------------------------
fitted = {name: (trans, cols) for name, trans, cols in pipeline.named_steps["preprocess"].transformers_}
ord_enc, ord_cols = fitted["ord"]
cat_enc, cat_cols = fitted["cat"]
num_enc, num_cols = fitted["num"]

ORD_OPTIONS = {c: [str(v) for v in cats] for c, cats in zip(ord_cols, ord_enc.categories_)}
CAT_OPTIONS = {c: [str(v) for v in cats] for c, cats in zip(cat_cols, cat_enc.categories_)}
NUM_MEAN = dict(zip(num_cols, num_enc.mean_))          # training average of every numeric column
ALL_FEATURES = list(ord_cols) + list(cat_cols) + list(num_cols)


# ----------------------------------------------------------------------
#  Friendly labels for the dropdowns (value sent to the model stays the code)
# ----------------------------------------------------------------------
QUAL = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical / Average", "Fa": "Fair", "Po": "Poor", "None": "None"}
NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "Greens": "Greens", "GrnHill": "Green Hills", "IDOTRR": "Iowa DOT & Rail Road",
    "Landmrk": "Landmark", "MeadowV": "Meadow Village", "Mitchel": "Mitchell", "NAmes": "North Ames",
    "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of Iowa State Univ.",
    "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}
ZONING = {"RL": "Residential – Low density", "RM": "Residential – Medium density", "RH": "Residential – High density",
          "FV": "Floating Village residential", "C (all)": "Commercial", "A (agr)": "Agriculture", "I (all)": "Industrial"}
SUBCLASS = {
    "20": "20 – 1-story, 1946 & newer", "30": "30 – 1-story, 1945 & older", "40": "40 – 1-story with finished attic",
    "45": "45 – 1½-story, unfinished", "50": "50 – 1½-story, finished", "60": "60 – 2-story, 1946 & newer",
    "70": "70 – 2-story, 1945 & older", "75": "75 – 2½-story", "80": "80 – Split / multi-level",
    "85": "85 – Split foyer", "90": "90 – Duplex", "120": "120 – 1-story PUD", "150": "150 – 1½-story PUD",
    "160": "160 – 2-story PUD", "180": "180 – PUD multi-level", "190": "190 – 2-family conversion",
}
BLDG = {"1Fam": "Single-family detached", "2fmCon": "Two-family conversion", "2FmCon": "Two-family conversion",
        "Duplex": "Duplex", "Twnhs": "Townhouse (inside unit)", "TwnhsI": "Townhouse (inside unit)",
        "TwnhsE": "Townhouse (end unit)"}
HOUSE_STYLE = {"1Story": "One story", "1.5Fin": "1½ story (finished)", "1.5Unf": "1½ story (unfinished)",
               "2Story": "Two story", "2.5Fin": "2½ story (finished)", "2.5Unf": "2½ story (unfinished)",
               "SFoyer": "Split foyer", "SLvl": "Split level"}
LOT_SHAPE = {"Reg": "Regular", "IR1": "Slightly irregular", "IR2": "Moderately irregular", "IR3": "Irregular"}
YES_NO = {"Y": "Yes", "N": "No"}
GARAGE_TYPE = {"Attchd": "Attached", "Detchd": "Detached", "BuiltIn": "Built-in", "Basment": "In basement",
               "CarPort": "Car port", "2Types": "More than one type", "None": "No garage"}
GARAGE_FINISH = {"Fin": "Finished", "RFn": "Rough finished", "Unf": "Unfinished", "None": "No garage"}
BSMT_EXPOSURE = {"Gd": "Good exposure", "Av": "Average exposure", "Mn": "Minimum exposure",
                 "No": "No exposure", "None": "No basement"}
BSMT_FIN_TYPE = {"GLQ": "Good living quarters", "ALQ": "Average living quarters", "BLQ": "Below-average living quarters",
                 "Rec": "Average rec room", "LwQ": "Low quality", "Unf": "Unfinished", "None": "No basement"}
ADV_LABELS = {"Exter Cond": QUAL, "Heating QC": QUAL,
              "Paved Drive": {"Y": "Paved", "P": "Partial pavement", "N": "Dirt / gravel"}}

# Typical values used for the "Advanced" fields (everything else falls back to the training average / first option)
DEFAULT_CAT = {
    "Street": "Pave", "Alley": "None", "Land Contour": "Lvl", "Lot Config": "Inside", "Condition 1": "Norm",
    "Condition 2": "Norm", "Roof Style": "Gable", "Roof Matl": "CompShg", "Exterior 1st": "VinylSd",
    "Exterior 2nd": "VinylSd", "Mas Vnr Type": "None", "Foundation": "PConc", "Heating": "GasA",
    "Electrical": "SBrkr", "Misc Feature": "None", "Sale Type": "WD", "Sale Condition": "Normal",
    "Utilities": "AllPub", "Exter Cond": "TA", "Heating QC": "Ex", "Paved Drive": "Y", "Land Slope": "Gtl",
    "Functional": "Typ", "Fence": "None",
}
DEFAULT_NUM = {"Mas Vnr Area": 0, "Enclosed Porch": 0, "3Ssn Porch": 0, "Screen Porch": 0, "Misc Val": 0}

# Columns that have their own friendly input above (so they are NOT repeated in "Advanced")
HANDLED = {
    "Neighborhood", "MS Zoning", "MS SubClass", "Bldg Type", "House Style", "Lot Area", "Lot Frontage", "Lot Shape",
    "1st Flr SF", "2nd Flr SF", "Gr Liv Area", "Low Qual Fin SF", "Bedroom AbvGr", "Full Bath", "Half Bath",
    "Kitchen AbvGr", "TotRms AbvGrd", "Fireplaces", "Fireplace Qu", "Overall Qual", "Overall Cond", "Year Built",
    "Year Remod/Add", "Exter Qual", "Kitchen Qual", "Central Air", "Total Bsmt SF", "BsmtFin SF 1", "BsmtFin SF 2",
    "Bsmt Unf SF", "Bsmt Full Bath", "Bsmt Half Bath", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1",
    "BsmtFin Type 2", "Garage Cars", "Garage Area", "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Garage Yr Blt", "Pool Area", "Pool QC", "Wood Deck SF", "Open Porch SF",
}


def choice(label, col, default=None, labels=None, exclude=(), help=None):
    """Dropdown whose options come from the trained pipeline."""
    opts = [o for o in (ORD_OPTIONS.get(col) or CAT_OPTIONS.get(col) or []) if o not in exclude]
    idx = 0
    if default is not None:
        for i, o in enumerate(opts):
            if o == default or o.strip() == str(default).strip():
                idx = i
                break
    fmt = (lambda v: labels.get(v, v)) if labels else (lambda v: v)
    return st.selectbox(label, opts, index=idx, format_func=fmt, help=help, key=f"w_{label}")


def money(x):
    return f"{CURRENCY}{int(x):,}"


# ----------------------------------------------------------------------
#  2. Front end
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero">
<h1>🏠 House Price Estimator</h1>
<p>Describe the property below and get an instant price range from a machine-learning model trained on historical sales.</p>
</div>
""", unsafe_allow_html=True)


with st.form("house_form"):
    t_prop, t_size, t_qual, t_extra, t_adv = st.tabs(
        ["📍 Property", "📐 Size & Rooms", "⭐ Quality & Age", "🚗 Garage, Basement & Extras", "⚙️ Advanced"])

    # ---------------- Property ----------------
    with t_prop:
        c1, c2 = st.columns(2)
        with c1:
            neighborhood = choice("Neighborhood", "Neighborhood", "NAmes", NEIGHBORHOODS)
            ms_zoning = choice("Zoning", "MS Zoning", "RL", ZONING)
            ms_subclass = choice("Dwelling type", "MS SubClass", "20", SUBCLASS)
            bldg_type = choice("Building type", "Bldg Type", "1Fam", BLDG)
        with c2:
            house_style = choice("House style", "House Style", "1Story", HOUSE_STYLE)
            lot_area = st.number_input("Lot area (sq ft)", min_value=1000, max_value=215000, value=9500, step=100)
            lot_frontage = st.number_input("Lot frontage (ft)", min_value=20, max_value=320, value=70, step=1)
            lot_shape = choice("Lot shape", "Lot Shape", "Reg", LOT_SHAPE)

    # ---------------- Size & rooms ----------------
    with t_size:
        c1, c2 = st.columns(2)
        with c1:
            first_flr = st.number_input("1st floor area (sq ft)", min_value=300, max_value=5000, value=1100, step=10)
            second_flr = st.number_input("2nd floor area (sq ft)", min_value=0, max_value=3000, value=0, step=10,
                                         help="Leave 0 for a single-storey house.")
            bedrooms = st.number_input("Bedrooms (above ground)", min_value=0, max_value=8, value=3)
            kitchens = st.number_input("Kitchens (above ground)", min_value=0, max_value=3, value=1)
        with c2:
            full_bath = st.number_input("Full bathrooms (above ground)", min_value=0, max_value=4, value=2)
            half_bath = st.number_input("Half bathrooms (above ground)", min_value=0, max_value=2, value=0)
            tot_rms = st.number_input("Total rooms above ground", min_value=2, max_value=15, value=6,
                                      help="Excludes bathrooms.")
            fireplaces = st.number_input("Fireplaces", min_value=0, max_value=4, value=0)

    # ---------------- Quality & age ----------------
    with t_qual:
        c1, c2 = st.columns(2)
        with c1:
            overall_qual = st.slider("Overall material & finish quality", 1, 10, 6,
                                     help="1 = Very poor … 5 = Average … 10 = Very excellent")
            overall_cond = st.slider("Overall condition", 1, 10, 5,
                                     help="1 = Very poor … 5 = Average … 10 = Very excellent")
            year_built = st.number_input("Year built", min_value=1870, max_value=2025, value=1975)
            year_remod = st.number_input("Year remodelled (same as built if never)", min_value=1950,
                                         max_value=2025, value=1995)
        with c2:
            exter_qual = choice("Exterior quality", "Exter Qual", "TA", QUAL)
            kitchen_qual = choice("Kitchen quality", "Kitchen Qual", "TA", QUAL)
            central_air = choice("Central air conditioning", "Central Air", "Y", YES_NO)

    # ---------------- Garage, basement, extras ----------------
    with t_extra:
        st.markdown("##### 🚗 Garage")
        has_garage = st.checkbox("Property has a garage", value=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            garage_cars = st.number_input("Garage capacity (cars)", min_value=1, max_value=5, value=2)
        with c2:
            garage_area = st.number_input("Garage area (sq ft)", min_value=100, max_value=1500, value=480, step=10)
        with c3:
            garage_type = choice("Garage type", "Garage Type", "Attchd", GARAGE_TYPE, exclude=("None",))
        with c4:
            garage_finish = choice("Garage interior", "Garage Finish", "Unf", GARAGE_FINISH, exclude=("None",))

        st.divider()
        st.markdown("##### 🧱 Basement")
        has_bsmt = st.checkbox("Property has a basement", value=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bsmt_fin = st.number_input("Finished area (sq ft)", min_value=0, max_value=3000, value=450, step=10)
            bsmt_full_bath = st.number_input("Basement full baths", min_value=0, max_value=3, value=0)
        with c2:
            bsmt_unf = st.number_input("Unfinished area (sq ft)", min_value=0, max_value=3000, value=550, step=10)
            bsmt_half_bath = st.number_input("Basement half baths", min_value=0, max_value=2, value=0)
        with c3:
            bsmt_qual = choice("Basement height quality", "Bsmt Qual", "TA", QUAL, exclude=("None",))
            bsmt_exposure = choice("Basement exposure", "Bsmt Exposure", "No", BSMT_EXPOSURE, exclude=("None",))
        with c4:
            bsmt_fin_type = choice("Finished area type", "BsmtFin Type 1", "GLQ", BSMT_FIN_TYPE, exclude=("None",))

        st.divider()
        st.markdown("##### 🌿 Outdoor & extras")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            wood_deck = st.number_input("Wood deck (sq ft)", min_value=0, max_value=1500, value=0, step=10)
        with c2:
            open_porch = st.number_input("Open porch (sq ft)", min_value=0, max_value=800, value=0, step=10)
        with c3:
            pool_area = st.number_input("Pool area (sq ft)", min_value=0, max_value=800, value=0, step=10)
            pool_qc = choice("Pool quality (if pool)", "Pool QC", "TA", QUAL, exclude=("None",))
        with c4:
            fireplace_qu = choice("Fireplace quality (if any)", "Fireplace Qu", "TA", QUAL, exclude=("None",))

    # ---------------- Advanced (auto-built from the trained pipeline) ----------------
    adv_values = {}
    with t_adv:
        st.caption("Less important details, pre-filled with typical values. Change them only if you know them.")
        adv_cols = [c for c in ALL_FEATURES if c not in HANDLED]
        grid = st.columns(3)
        for i, col in enumerate(adv_cols):
            with grid[i % 3]:
                if col in ORD_OPTIONS or col in CAT_OPTIONS:
                    adv_values[col] = choice(col, col, DEFAULT_CAT.get(col), ADV_LABELS.get(col))
                else:
                    default = DEFAULT_NUM.get(col, NUM_MEAN.get(col, 0))
                    adv_values[col] = st.number_input(col, min_value=0, value=int(round(default)), step=1,
                                                      key=f"w_{col}")

    submitted = st.form_submit_button("💰 Get price estimate", use_container_width=True, type="primary")


# ----------------------------------------------------------------------
#  3 - 6. Build the input row, predict, show the range
# ----------------------------------------------------------------------
if submitted:
    if year_remod < year_built:
        st.error("Year remodelled can't be earlier than the year built.")
        st.stop()
    if tot_rms < bedrooms + kitchens:
        st.warning("Total rooms above ground is smaller than bedrooms + kitchens – please double-check.")

    gr_liv_area = first_flr + second_flr

    # This ONE row must have exactly the same column names the model was trained on.
    row = {
        "Neighborhood": neighborhood, "MS Zoning": ms_zoning, "MS SubClass": ms_subclass,
        "Bldg Type": bldg_type, "House Style": house_style,
        "Lot Area": lot_area, "Lot Frontage": lot_frontage, "Lot Shape": lot_shape,
        "1st Flr SF": first_flr, "2nd Flr SF": second_flr, "Gr Liv Area": gr_liv_area, "Low Qual Fin SF": 0,
        "Bedroom AbvGr": bedrooms, "Kitchen AbvGr": kitchens, "Full Bath": full_bath, "Half Bath": half_bath,
        "TotRms AbvGrd": tot_rms, "Fireplaces": fireplaces,
        "Fireplace Qu": fireplace_qu if fireplaces > 0 else "None",
        "Overall Qual": overall_qual, "Overall Cond": overall_cond,
        "Year Built": year_built, "Year Remod/Add": year_remod,
        "Exter Qual": exter_qual, "Kitchen Qual": kitchen_qual, "Central Air": central_air,
        "Wood Deck SF": wood_deck, "Open Porch SF": open_porch,
        "Pool Area": pool_area, "Pool QC": pool_qc if pool_area > 0 else "None",
    }

    if has_garage:
        row.update({"Garage Cars": garage_cars, "Garage Area": garage_area, "Garage Type": garage_type,
                    "Garage Finish": garage_finish, "Garage Qual": "TA", "Garage Cond": "TA",
                    "Garage Yr Blt": year_built})
    else:
        row.update({"Garage Cars": 0, "Garage Area": 0, "Garage Type": "None", "Garage Finish": "None",
                    "Garage Qual": "None", "Garage Cond": "None", "Garage Yr Blt": 0})

    if has_bsmt:
        row.update({"Total Bsmt SF": bsmt_fin + bsmt_unf, "BsmtFin SF 1": bsmt_fin, "BsmtFin SF 2": 0,
                    "Bsmt Unf SF": bsmt_unf, "Bsmt Full Bath": bsmt_full_bath, "Bsmt Half Bath": bsmt_half_bath,
                    "Bsmt Qual": bsmt_qual, "Bsmt Cond": "TA", "Bsmt Exposure": bsmt_exposure,
                    "BsmtFin Type 1": bsmt_fin_type, "BsmtFin Type 2": "Unf"})
    else:
        row.update({"Total Bsmt SF": 0, "BsmtFin SF 1": 0, "BsmtFin SF 2": 0, "Bsmt Unf SF": 0,
                    "Bsmt Full Bath": 0, "Bsmt Half Bath": 0, "Bsmt Qual": "None", "Bsmt Cond": "None",
                    "Bsmt Exposure": "None", "BsmtFin Type 1": "None", "BsmtFin Type 2": "None"})

    row.update(adv_values)

    # The model expects every column it saw in training (in the same order)
    expected_cols = list(pipeline.feature_names_in_)
    for c in expected_cols:
        row.setdefault(c, 0)
    input_row = pd.DataFrame([row])[expected_cols]

    predicted_price = max(0.0, float(pipeline.predict(input_row)[0]))
    low_price = max(0.0, predicted_price - rmse)
    high_price = predicted_price + rmse

    # Round to the nearest thousand -- a price like $187,283 is false precision;
    # $187,000 is what a real listing would show.
    def round_to_thousand(x):
        return int(round(x / 1000.0) * 1000)

    low_price = round_to_thousand(low_price)
    high_price = round_to_thousand(high_price)
    predicted_price_rounded = round_to_thousand(predicted_price)

    st.markdown(f"""
<div class="result-card">
<div class="result-label">Estimated price range</div>
<div class="result-range">{money(low_price)} – {money(high_price)}</div>
<div class="result-sub">Best estimate <b>{money(predicted_price_rounded)}</b> · typical error ± {money(round_to_thousand(rmse))}</div>
</div>
<div class="bar-wrap">
<div class="bar"><div class="bar-marker"></div></div>
<div class="bar-labels"><span>Low · {money(low_price)}</span><span>Best estimate · {money(predicted_price_rounded)}</span><span>High · {money(high_price)}</span></div>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Living area (above ground)", f"{gr_liv_area:,} sq ft")
    m2.metric("Estimated price per sq ft", f"{CURRENCY}{predicted_price / max(gr_liv_area, 1):,.0f}")
    m3.metric("Property age", f"{max(0, 2025 - year_built)} years")

    st.caption(f"Based on a Linear Regression model trained on historical sales. "
               f"Typical error {money(round_to_thousand(rmse))}.")

    with st.expander("See the exact input sent to the model"):
        st.dataframe(input_row.T.rename(columns={0: "Value"}).astype(str), use_container_width=True)