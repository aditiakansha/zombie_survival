import pandas as pd
from pathlib import Path


DEMO_DIR = Path("data/raw/nhanes")
MORTALITY_DIR = Path("data/raw/mortality")


DEMO_FILES = [
    "DEMO.XPT",
    "DEMO_B.XPT",
    "DEMO_C.XPT",
    "DEMO_D.XPT",
    "DEMO_E.XPT",
    "DEMO_F.XPT",
    "DEMO_G.XPT",
    "DEMO_H.XPT",
    "DEMO_I.XPT",
    "DEMO_J.XPT",
]


MORTALITY_FILES = [
    "NHANES_1999_2000_MORT_2019_PUBLIC.dat",
    "NHANES_2001_2002_MORT_2019_PUBLIC.dat",
    "NHANES_2003_2004_MORT_2019_PUBLIC.dat",
    "NHANES_2005_2006_MORT_2019_PUBLIC.dat",
    "NHANES_2007_2008_MORT_2019_PUBLIC.dat",
    "NHANES_2009_2010_MORT_2019_PUBLIC.dat",
    "NHANES_2011_2012_MORT_2019_PUBLIC.dat",
    "NHANES_2013_2014_MORT_2019_PUBLIC.dat",
    "NHANES_2015_2016_MORT_2019_PUBLIC.dat",
    "NHANES_2017_2018_MORT_2019_PUBLIC.dat",
]


def load_demographics():
    dataframes = []

    for filename in DEMO_FILES:
        file_path = DEMO_DIR / filename

        df = pd.read_sas(file_path)

        dataframes.append(df)

    return pd.concat(
        dataframes,
        ignore_index=True,
        sort=False
    )


def load_mortality():
    dataframes = []

    for filename in MORTALITY_FILES:
        file_path = MORTALITY_DIR / filename

        df = pd.read_fwf(
            file_path,
            colspecs=[
                (0, 14),   # SEQN
                (14, 15),  # ELIGSTAT
                (15, 16),  # MORTSTAT
            ],
            names=[
                "SEQN",
                "ELIGSTAT",
                "MORTSTAT",
            ],
            dtype=str,
        )

        dataframes.append(df)

    return pd.concat(
        dataframes,
        ignore_index=True
    )


def build_dataset():

    demographics = load_demographics()
    mortality = load_mortality()

    # Convert SEQN to the same numeric representation
    demographics["SEQN"] = pd.to_numeric(
        demographics["SEQN"],
        errors="coerce"
    )

    mortality["SEQN"] = pd.to_numeric(
        mortality["SEQN"],
        errors="coerce"
    )

    # Convert eligibility and mortality status
    mortality["ELIGSTAT"] = pd.to_numeric(
        mortality["ELIGSTAT"],
        errors="coerce"
    )

    mortality["MORTSTAT"] = pd.to_numeric(
        mortality["MORTSTAT"],
        errors="coerce"
    )

    # Keep only participants eligible for public-use mortality follow-up
    mortality = mortality[
        mortality["ELIGSTAT"] == 1
    ].copy()

    # Merge using SEQN
    merged = demographics.merge(
        mortality,
        on="SEQN",
        how="inner",
        validate="one_to_one"
    )

    return merged


if __name__ == "__main__":

    dataset = build_dataset()

    print("=" * 70)
    print("FINAL INITIAL DATASET")
    print("=" * 70)

    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    print("\nMortality outcome:")
    print(dataset["MORTSTAT"].value_counts(dropna=False))

    print("\nMissing MORTSTAT:")
    print(dataset["MORTSTAT"].isna().sum())

    print("\nFirst 5 rows:")
    print(dataset.head())