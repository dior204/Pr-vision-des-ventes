# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 16:03:00 2026

@author: HP
"""

# utils/pipeline.py
import pandas as pd
import numpy as np
import gc


class FavoritaFeaturePipeline:
    """
    Pipeline Fit/Transform pour créer des features Favorita
    - anti-leakage (fit sur historique, transform sur futur)
    - robuste en production (fallbacks)
    - compatible split temporel avec GAP (train historique seulement)
    """

    def __init__(self, data_dir, sales_history_days=120, verbose=True):
        self.data_dir = data_dir
        self.sales_history_days = sales_history_days
        self.verbose = verbose

        # appris au fit()
        self.sales_target_col = None
        self.store_freq = None
        self.item_freq = None

        self.items_fe = None
        self.store_maps = {}

        self.trans_s = None
        self.last_roll_by_store = None

        self.oil_s = None
        self.oil_fallback = 0.0

        self.holiday_nat_maps = {}
        self.holiday_override_maps = {}

        # lookup ventes (train uniquement)
        self.sales_s = None

    # ------------------------
    # helpers
    # ------------------------
    def _log(self, msg):
        if self.verbose:
            print(msg)

    @staticmethod
    def _ensure_datetime(df, col="date"):
        if df[col].dtype != "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    @staticmethod
    def _force_int(df, col, dtype):
        # évite Categorical -> fillna(0) qui plante
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("int64")
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
        return df

    @staticmethod
    def _make_key_store_date(store_series, date_series):
        ymd = date_series.dt.year * 10000 + date_series.dt.month * 100 + date_series.dt.day
        return (store_series.astype("int32") * 1_000_000 + ymd.astype("int32")).astype("int64")

    # ------------------------
    # FIT
    # ------------------------
    def fit(self, history_df: pd.DataFrame):
        """
        history_df = train_fit uniquement (pas gap, pas test)
        Doit contenir : date, store_nbr, item_nbr + cible (unit_sales_log ou unit_sales_clean)
        """
        df = history_df.copy()
        df = self._ensure_datetime(df, "date")

        # types safe
        df = self._force_int(df, "store_nbr", "int16")
        df = self._force_int(df, "item_nbr", "int32")

        # cible
        if "unit_sales_log" in df.columns:
            self.sales_target_col = "unit_sales_log"
        elif "unit_sales_clean" in df.columns:
            self.sales_target_col = "unit_sales_clean"
        else:
            raise ValueError("Il faut unit_sales_log ou unit_sales_clean dans history_df")

        self._log("🔧 FIT : fréquences + lookups + tables externes...")

        # freq enc sur historique uniquement
        self.store_freq = df["store_nbr"].value_counts(normalize=True).astype("float32")
        self.item_freq = df["item_nbr"].value_counts(normalize=True).astype("float32")

        # sales history (limité)
        max_d = df["date"].max()
        min_keep = max_d - pd.Timedelta(days=int(self.sales_history_days))
        hs = df.loc[df["date"] >= min_keep, ["store_nbr", "item_nbr", "date", self.sales_target_col]].copy()

        hs["store_nbr"] = hs["store_nbr"].astype("int16")
        hs["item_nbr"] = hs["item_nbr"].astype("int32")
        hs[self.sales_target_col] = pd.to_numeric(hs[self.sales_target_col], errors="coerce").fillna(0).astype("float32")
        hs = hs.drop_duplicates(["store_nbr", "item_nbr", "date"], keep="last")

        self.sales_s = hs.set_index(["store_nbr", "item_nbr", "date"])[self.sales_target_col]
        del hs
        gc.collect()

        # ---------------- ITEMS ----------------
        items = pd.read_csv(f"{self.data_dir}/items.csv")
        items["item_nbr"] = pd.to_numeric(items["item_nbr"], errors="coerce").fillna(0).astype("int32")
        items["family"] = items["family"].fillna("UNKNOWN").astype(str).str.strip()
        items["class"] = items["class"].fillna(-1).astype("int16").astype(str)
        items["perishable"] = items["perishable"].fillna(0).astype("int8")

        fam_freq = items["family"].value_counts(normalize=True)
        cls_freq = items["class"].value_counts(normalize=True)
        items["family_freq_items"] = items["family"].map(fam_freq).fillna(0).astype("float32")
        items["class_freq_items"] = items["class"].map(cls_freq).fillna(0).astype("float32")

        pair_counts = items.groupby(["family", "class"]).size()
        family_counts = items.groupby("family").size()
        items["class_freq_in_family"] = (
            items.set_index(["family", "class"]).index.map(pair_counts) /
            items["family"].map(family_counts).to_numpy()
        ).astype("float32")

        self.items_fe = items[[
            "item_nbr", "perishable", "family_freq_items", "class_freq_items", "class_freq_in_family"
        ]].drop_duplicates("item_nbr")

        del items
        gc.collect()

        # ---------------- STORES ----------------
        stores = pd.read_csv(f"{self.data_dir}/stores.csv")
        stores["store_nbr"] = pd.to_numeric(stores["store_nbr"], errors="coerce").fillna(0).astype("int16")
        stores["type"] = stores["type"].fillna("UNKNOWN").astype(str).str.strip()
        stores["state"] = stores["state"].fillna("UNKNOWN").astype(str).str.strip()
        stores["city"] = stores["city"].fillna("UNKNOWN").astype(str).str.strip()
        stores["cluster"] = stores["cluster"].fillna(-1).astype("int16")

        type_freq = stores["type"].value_counts(normalize=True)
        state_freq = stores["state"].value_counts(normalize=True)
        city_freq = stores["city"].value_counts(normalize=True)

        stores["type_freq"] = stores["type"].map(type_freq).fillna(0).astype("float32")
        stores["state_freq"] = stores["state"].map(state_freq).fillna(0).astype("float32")
        stores["city_freq"] = stores["city"].map(city_freq).fillna(0).astype("float32")

        pair_counts = stores.groupby(["state", "city"]).size()
        state_counts = stores.groupby("state").size()
        stores["city_freq_in_state"] = (
            stores.set_index(["state", "city"]).index.map(pair_counts) /
            stores["state"].map(state_counts).to_numpy()
        ).astype("float32")

        self.store_maps = {
            "cluster": stores.set_index("store_nbr")["cluster"].to_dict(),
            "type_freq": stores.set_index("store_nbr")["type_freq"].to_dict(),
            "state_freq": stores.set_index("store_nbr")["state_freq"].to_dict(),
            "city_freq": stores.set_index("store_nbr")["city_freq"].to_dict(),
            "city_freq_in_state": stores.set_index("store_nbr")["city_freq_in_state"].to_dict(),
        }

        del stores
        gc.collect()

        # ---------------- TRANSACTIONS roll14 ----------------
        tr = pd.read_csv(f"{self.data_dir}/transactions.csv", parse_dates=["date"])
        tr["store_nbr"] = pd.to_numeric(tr["store_nbr"], errors="coerce").fillna(0).astype("int16")
        tr["transactions"] = pd.to_numeric(tr["transactions"], errors="coerce").fillna(0).astype("float32")

        tr = (tr.groupby(["store_nbr", "date"], as_index=False)
                .agg(transactions=("transactions", "sum"))
                .sort_values(["store_nbr", "date"]))

        tr["transactions_roll14"] = (
            tr.groupby("store_nbr")["transactions"]
              .transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
        ).astype("float32")

        self.trans_s = tr.set_index(["store_nbr", "date"])["transactions_roll14"]
        self.last_roll_by_store = tr.groupby("store_nbr")["transactions_roll14"].last().astype("float32")

        del tr
        gc.collect()

        # ---------------- OIL roll14 ----------------
        oil = pd.read_csv(f"{self.data_dir}/oil.csv", parse_dates=["date"]).sort_values("date")
        oil["dcoilwtico"] = pd.to_numeric(oil["dcoilwtico"], errors="coerce").ffill()

        oil["oil_roll14"] = oil["dcoilwtico"].shift(1).rolling(14, min_periods=1).mean().astype("float32")
        self.oil_s = oil.set_index("date")["oil_roll14"]

        self.oil_fallback = float(self.oil_s.tail(30).mean()) if len(self.oil_s) else 0.0

        del oil
        gc.collect()

        # ---------------- HOLIDAYS ----------------
        hol = pd.read_csv(f"{self.data_dir}/holidays_events.csv")
        hol["date"] = pd.to_datetime(hol["date"], errors="coerce")
        for col in ["type", "locale", "locale_name", "description"]:
            hol[col] = hol[col].fillna("").astype(str).str.strip()
        hol["transferred"] = hol["transferred"].fillna(False).astype(int)

        hol["event_key"] = hol["description"] + " | " + hol["locale"] + " | " + hol["locale_name"]
        transfer_map = (
            hol.loc[hol["type"] == "Transfer", ["event_key", "date"]]
              .drop_duplicates("event_key")
              .set_index("event_key")["date"]
              .to_dict()
        )

        hol["observed_date"] = hol["date"]
        mask_moved = (hol["type"] == "Holiday") & (hol["transferred"] == 1)
        hol.loc[mask_moved, "observed_date"] = hol.loc[mask_moved, "event_key"].map(transfer_map)
        hol["observed_date"] = hol["observed_date"].fillna(hol["date"])

        hol["f_holiday"] = (hol["type"] == "Holiday").astype("int8")

        # national maps
        national = (hol[hol["locale"] == "National"]
                    .groupby("observed_date", as_index=False)
                    .agg(nat_h=("f_holiday", lambda s: int(s.sum() > 0)))
                    .rename(columns={"observed_date": "date"}))

        self.holiday_nat_maps = {"nat_h": dict(zip(national["date"], national["nat_h"]))}

        # local/regional overrides
        stores_loc = pd.read_csv(f"{self.data_dir}/stores.csv")
        stores_loc["store_nbr"] = pd.to_numeric(stores_loc["store_nbr"], errors="coerce").fillna(0).astype("int16")
        stores_loc["city"] = stores_loc["city"].fillna("UNKNOWN").astype(str).str.strip()
        stores_loc["state"] = stores_loc["state"].fillna("UNKNOWN").astype(str).str.strip()
        store_loc = stores_loc[["store_nbr", "city", "state"]].drop_duplicates("store_nbr")

        regional = hol[hol["locale"] == "Regional"].copy()
        regional["date"] = regional["observed_date"]
        regional = regional.drop(columns=["observed_date"])
        regional = regional.merge(store_loc, left_on="locale_name", right_on="state", how="inner")
        regional = (regional.groupby(["date", "store_nbr"], as_index=False)
                    .agg(h=("f_holiday", lambda s: int(s.sum() > 0))))

        local = hol[hol["locale"] == "Local"].copy()
        local["date"] = local["observed_date"]
        local = local.drop(columns=["observed_date"])
        local = local.merge(store_loc, left_on="locale_name", right_on="city", how="inner")
        local = (local.groupby(["date", "store_nbr"], as_index=False)
                 .agg(h=("f_holiday", lambda s: int(s.sum() > 0))))

        over = pd.concat([regional, local], ignore_index=True).groupby(["date", "store_nbr"], as_index=False).max()
        over["key"] = self._make_key_store_date(over["store_nbr"], over["date"])
        self.holiday_override_maps = {"h": dict(zip(over["key"], over["h"]))}

        del hol, stores_loc, store_loc, national, regional, local, over
        gc.collect()

        self._log("✅ FIT terminé.")
        return self

    # ------------------------
    # TRANSFORM
    # ------------------------
    def transform(self, df_new: pd.DataFrame):
        """
        df_new peut être gap/test/nouvelles données (prod/unit test)
        colonnes minimales : date, store_nbr, item_nbr (+ onpromotion optionnel)
        """
        df = df_new.copy()
        df = self._ensure_datetime(df, "date")

        # types safe
        df = self._force_int(df, "store_nbr", "int16")
        df = self._force_int(df, "item_nbr", "int32")

        # ---------------- date feats
        d = df["date"]
        df["year"] = d.dt.year.astype("int16")
        df["month"] = d.dt.month.astype("int8")
        df["day"] = d.dt.day.astype("int8")
        df["dow"] = d.dt.dayofweek.astype("int8")
        df["is_weekend"] = (df["dow"] >= 5).astype("int8")

        # ---------------- promo
        if "onpromotion" in df.columns:
            # force bool
            df["onpromotion"] = df["onpromotion"].astype("bool")
            df["onpromo"] = df["onpromotion"].astype("int8")
        else:
            df["onpromo"] = 0

        # ---------------- freq enc
        # Important: map renvoie float, donc fillna(0) safe
        df["store_freq"] = df["store_nbr"].map(self.store_freq).astype("float32")
        df["item_freq"] = df["item_nbr"].map(self.item_freq).astype("float32")
        df["store_freq"] = df["store_freq"].fillna(0).astype("float32")
        df["item_freq"] = df["item_freq"].fillna(0).astype("float32")

        # ---------------- items merge
        df = df.merge(self.items_fe, on="item_nbr", how="left")
        for c in ["perishable", "family_freq_items", "class_freq_items", "class_freq_in_family"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

        df["perishable"] = df["perishable"].astype("int8")

        # ---------------- store maps
        df["cluster"] = df["store_nbr"].map(self.store_maps["cluster"]).fillna(-1).astype("int16")
        df["type_freq"] = df["store_nbr"].map(self.store_maps["type_freq"]).fillna(0).astype("float32")
        df["state_freq"] = df["store_nbr"].map(self.store_maps["state_freq"]).fillna(0).astype("float32")
        df["city_freq"] = df["store_nbr"].map(self.store_maps["city_freq"]).fillna(0).astype("float32")
        df["city_freq_in_state"] = df["store_nbr"].map(self.store_maps["city_freq_in_state"]).fillna(0).astype("float32")

        # ---------------- transactions_roll14
        keys = pd.MultiIndex.from_frame(df[["store_nbr", "date"]])
        tr_vals = self.trans_s.reindex(keys).to_numpy(dtype="float32")
        df["transactions_roll14"] = tr_vals
        df["transactions_roll14"] = (
            pd.Series(df["transactions_roll14"])
              .fillna(df["store_nbr"].map(self.last_roll_by_store))
              .fillna(0)
              .astype("float32")
        )

        # ---------------- oil_roll14
        df["oil_roll14"] = self.oil_s.reindex(df["date"]).to_numpy(dtype="float32")
        df["oil_roll14"] = pd.Series(df["oil_roll14"]).fillna(self.oil_fallback).astype("float32")

        # ---------------- holiday effective
        nat_h = self.holiday_nat_maps["nat_h"]
        df["is_holiday_effective"] = df["date"].map(nat_h).fillna(0).astype("int8")

        k = self._make_key_store_date(df["store_nbr"], df["date"])
        h_over = pd.Series(k).map(self.holiday_override_maps["h"]).fillna(0).to_numpy()
        df["is_holiday_effective"] = np.maximum(df["is_holiday_effective"].to_numpy(), h_over).astype("int8")

        # ==================================================
        # SALES LAG28 + ROLL14 (anti-leak, lookup train only)
        # ==================================================
        # ---- Lag 28
        lag_idx = pd.MultiIndex.from_frame(pd.DataFrame({
            "store_nbr": df["store_nbr"].astype("int16"),
            "item_nbr": df["item_nbr"].astype("int32"),
            "date": df["date"] - pd.Timedelta(days=28)
        }))
        lag_vals = self.sales_s.reindex(lag_idx).to_numpy(dtype="float32")
        df["sales_lag28"] = pd.Series(lag_vals).fillna(0).astype("float32")

        # ---- Roll 14 + cnt (nb jours dispo)
        W = 14
        acc = np.zeros(len(df), dtype="float32")
        cnt = np.zeros(len(df), dtype="float32")

        for j in range(1, W + 1):  # t-1 ... t-14
            r_idx = pd.MultiIndex.from_frame(pd.DataFrame({
                "store_nbr": df["store_nbr"].astype("int16"),
                "item_nbr": df["item_nbr"].astype("int32"),
                "date": df["date"] - pd.Timedelta(days=j)
            }))
            vals = self.sales_s.reindex(r_idx).to_numpy(dtype="float32")
            mask = ~pd.isna(vals)
            acc[mask] += vals[mask]
            cnt[mask] += 1.0

        df["sales_cnt14"] = cnt.astype("float32")
        df["sales_roll14"] = np.where(cnt > 0, acc / cnt, 0.0).astype("float32")

        return df