from typing import Any, Tuple
import logging
from datetime import date, timedelta
import math
import uuid
import random
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from faker import Faker
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from luntaiDs.ModelingTools.FeatureEngineer.transformers import NamedTransformer
from luntaiDs.ModelingTools.FeatureEngineer.preprocessing import floatFy, nominal_categ_preprocess_pipe, \
    numeric_preprocess_pipe, ordinal_categ_preprocess_pipe, binary_preprocess_pipe
from luntaiDs.ModelingTools.FeatureEngineer.transformers import WeightedAverager
from luntaiDs.ModelingTools.utils.support import make_present_col_selector
from luntaiDs.ModelingTools.Explore.profiling import TabularStat

fake = Faker()

class CustFeature:
    # generate a map of real office to fake office
    offices = ['New York', 'Austin', 'Seattle', 'Chicago']
    # codify the hierarchical structure
    allowed_orgs_per_office = {
        'New York': ['Sales'],
        'Austin': ['Devops', 'Platform', 'Product', 'Internal Tools'],
        'Chicago': ['Devops'], 'Seattle': ['Internal Tools', 'Product']
    }
    allowed_titles_per_org = {
        'Devops': ['Engineer', 'Senior Engineer', 'Manager'],
        'Sales': ['Associate'],
        'Platform': ['Engineer'],
        'Product': ['Manager', 'VP'],
        'Internal Tools': ['Engineer', 'Senior Engineer', 'VP', 'Manager']
    }
    title_and_salary_range = {
        'Engineer': [90, 120], 'Senior Engineer': [110, 140],
        'Manager': [130, 150], 'Associate': [60, 80],
        'VP': [150, 250]
    }

    @classmethod
    def new(cls, snap_dt: date) -> dict:
        # generate gender
        fak = fake.profile(fields = [
            'name', 'sex', 'ssn', 'mail',
            'blood_group', 'job', 'address'
        ])
        profile = {}
        profile['NAME'] = fak['name']
        profile['GENDER'] = fak['sex']
        profile['PHONE'] = fak['ssn']
        profile['EMAIL'] = fak['mail']
        profile['BLOOD_GRP'] = fak['blood_group']
        profile['JOB'] = fak['job']
        profile['ADDRESS'] = fak['address']
        profile['CUST_ID'] = uuid.uuid4().fields[-1] // 10000
        profile['OFFICE'] = random.choice(cls.offices)
        profile['ORG'] = random.choice(cls.allowed_orgs_per_office[profile['OFFICE']])
        profile['TITLE'] = random.choice(cls.allowed_titles_per_org[profile['ORG']])
        salary_range = cls.title_and_salary_range[profile['TITLE']]
        profile['SALARY'] = round(random.randint(
            1000 * salary_range[0],
            1000 * salary_range[1]
        ) / 1000) * 1000
        bonus_ratio = random.uniform(0.15, 0.2)
        profile['BONUS'] = round(profile['SALARY'] * bonus_ratio / 500) * 500
        st = fake.date_between(start_date=date(2014, 1, 1), end_date=snap_dt)
        bt = st - timedelta(days=365 * random.randint(1, 40))
        profile['BIRTH_DT'] = bt.strftime('%Y-%m-%d')
        profile['SINCE_DT'] = st.strftime('%Y-%m-%d')
        profile['SNAP_DT'] = snap_dt.strftime('%Y-%m-%d')
        return profile

    @classmethod
    def variate(cls, snap_dt: date, *pasts: dict) -> dict:
        last1 = pasts[0]
        new = last1.copy()
        new['SNAP_DT'] = snap_dt.strftime('%Y-%m-%d')
        if snap_dt.month == 12:
            new['SALARY'] = round(last1['SALARY'] * random.randint(90, 130) / 100)
            new['BONUS'] = round(last1['bonus'] * random.randint(30, 130) / 100)
        if random.randint(0, 15) < 2:
            new['OFFICE'] = random.choice(cls.offices)
            new['ORG'] = random.choice(cls.allowed_orgs_per_office[new['OFFICE']])
            new['TITLE'] = random.choice(cls.allowed_titles_per_org[new['ORG']])
            salary_range = cls.title_and_salary_range[new['TITLE']]
            new['SALARY'] = round(random.randint(
                1000 * salary_range[0],
                1000 * salary_range[1]
            ) / 1000) * 1000
            bonus_ratio = random.uniform(0.15, 0.2)
            new['BONUS'] = round(new['SALARY'] * bonus_ratio / 500) * 500
        return new

def lognormal2normal(mu: float, sigma: float) -> Tuple[float, float]:
    # https://en.wikipedia.org/wiki/Log-normal_distribution
    mu_ = math.log(mu ** 2 / (sigma ** 2 + mu ** 2) ** 0.5)
    sigma_ = (math.log(1 + sigma ** 2 / mu ** 2)) ** 0.5
    return mu_, sigma_

class AcctFeature:
    ACCT_TYPE_CDS = {'CHQ' : 0.3, 'SAV' : 0.1, 'CRD' : 0.4, 'LN': 0.05, 'MTG': 0.15}
    BAL_DISTR = {
        'CHQ': {
            'END_BAL' : {'mu' : 1000, 'sigma' : 500},
            'DEBIT' : {'a' : 3, 'b' :  5},
            'CREDIT' : {'a' : 3, 'b' : 5},
            'DIRECTION' : 1
        },
        'SAV': {
            'END_BAL' : {'mu' : 8000, 'sigma' : 2500},
            'DEBIT' : {'a' : 1, 'b' :  10},
            'CREDIT' : {'a' : 0.5, 'b' : 25},
            'DIRECTION' : 1
        },
        'CRD': {
            'END_BAL' : {'mu' : 1200, 'sigma' : 1000},
            'DEBIT' : {'a' : 5, 'b' :  5},
            'CREDIT' : {'a' : 5, 'b' : 5},
            'DIRECTION' : -1
        },
        'LN': {
            'END_BAL' : {'mu' : 10000, 'sigma' : 4000},
            'DEBIT' : {'a' : 1, 'b' :  45},
            'CREDIT' : {'a' : 0.1, 'b' : 45},
            'DIRECTION' : -1
        },
        'MTG': {
            'END_BAL' : {'mu' : 250000, 'sigma' : 100000},
            'DEBIT' : {'a' : 1, 'b' :  55},
            'CREDIT' : {'a' : 0.1, 'b' : 55},
            'DIRECTION' : -1
        }
    }

    @classmethod
    def new(cls, snap_dt: date) -> dict:
        acct = {}
        acct['ACCT_ID'] = str(uuid.uuid4())[:13]
        acct['SNAP_DT'] = snap_dt.strftime('%Y-%m-%d')
        acct['ACCT_TYPE_CD'] = random.choices(
            population=list(cls.ACCT_TYPE_CDS.keys()),
            weights=list(cls.ACCT_TYPE_CDS.values()),
            k = 1
        )[0]
        mu, sigma = lognormal2normal(
            mu=cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['END_BAL']['mu'],
            sigma=cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['END_BAL']['sigma'],
        )
        begin_bal = round(random.lognormvariate(mu, sigma), 2)
        debit_ratio = random.betavariate(
            alpha=cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['DEBIT']['a'],
            beta=cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['DEBIT']['b'],
        )
        credit_ratio = random.betavariate(
            alpha=cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['CREDIT']['a'],
            beta=cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['CREDIT']['b'],
        )
        acct['DR_AMT'] = round(debit_ratio * begin_bal, 2)
        acct['CR_AMT'] = round(credit_ratio * begin_bal, 2)
        acct['END_BAL'] = round(begin_bal + cls.BAL_DISTR[acct['ACCT_TYPE_CD']]['DIRECTION'] * (
            acct['DR_AMT'] - acct['CR_AMT']
        ), 2)
        if acct['ACCT_TYPE_CD'] in ['CRD', 'LN', 'MTG']:
            acct['CR_LMT'] = round((random.random() + 0.95) * max(acct['END_BAL'], begin_bal), 2)
        else:
            acct['CR_LMT'] = None
        return acct

    @classmethod
    def variate(cls, snap_dt: date, *pasts: dict) -> dict:
        # must have at least 1 past record
        last1 = pasts[0]
        new = last1.copy()
        new['SNAP_DT'] = snap_dt.strftime('%Y-%m-%d')
        begin_bal = new['END_BAL']
        new['DR_AMT'] = round(new['DR_AMT'] * (0.5 + 0.7 * random.random()), 2)
        new['CR_AMT'] = round(new['CR_AMT'] * (0.5 + 0.7 * random.random()), 2)
        new['END_BAL'] = round(begin_bal + cls.BAL_DISTR[new['ACCT_TYPE_CD']]['DIRECTION'] * (
                new['DR_AMT'] - new['CR_AMT']
        ), 2)
        # for credit card, have 5% chance increasing credit limit
        if new['ACCT_TYPE_CD'] in ['CRD']:
            new['CR_LMT'] = round(new['CR_LMT'] * (
                    1 +
                    (random.random() < 0.05) * random.betavariate(1, 5)
            ), 2)

        return new

class FakeLinearProbModel:
    def __init__(self, profile: dict, model_coeffs: dict):
        self.profile = profile
        self.model_coeffs = model_coeffs
        
    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        ts = TabularStat.from_dict(self.profile)

        transformers = []
        for col, stat in ts.configs.items():
            if col in ts.get_nominal_cols():
                transformer = nominal_categ_preprocess_pipe(
                    cs=stat,
                    impute_value='Missing',
                    bucket_strategy='freq',
                    encode_strategy='ohe'
                )
            elif col in ts.get_numeric_cols():
                transformer = numeric_preprocess_pipe(
                    ns=stat,
                    impute=True,
                    normalize=False,
                    standardize_strategy='robust'
                )
            elif col in ts.get_binary_cols():
                transformer = binary_preprocess_pipe(
                    bs=stat
                )
            elif col in ts.get_ordinal_cols():
                transformer = ordinal_categ_preprocess_pipe(
                    os=stat,
                    impute=True,
                    standardize=True
                )
            else:
                raise

            transformers.append((col, transformer, make_present_col_selector([col])))

        preprocessing_pipe = Pipeline([
            ('transform', NamedTransformer(
                ColumnTransformer(
                    transformers=transformers,
                    remainder='drop'
                )
            )),
            ('float', NamedTransformer(FunctionTransformer(floatFy)))
        ])
        preprocessed = preprocessing_pipe.fit_transform(features_df)

        intercept = self.model_coeffs['INTERCEPT']
        coeffs = {}
        for k, v in self.model_coeffs['COEF'].items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    k_ = k + "_" + sub_k
                    coeffs[k_] = sub_v
            else:
                coeffs[k] = v
        coeffs = pd.Series(coeffs)

        preprocessed = preprocessed[coeffs.index]
        model = WeightedAverager(
            weights=coeffs.array,
            intercept=intercept
        )
        score = model.fit_transform(preprocessed)
        return pd.Series(
            expit(score[:, 0] + self.model_coeffs['RANDOMNESS'] * np.random.random(len(score))),
            index = features_df.index
        )