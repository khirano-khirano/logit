# views.py
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
def index(request):
    if request.method == 'POST':
        analysis_type = request.POST.get('analysis_type')
        csv_file = request.FILES['csv_file']
        df = pd.read_csv(csv_file,encoding="utf-8")
        column_names = list(df)
        object = column_names[0]
        if analysis_type == 'logistic':
             # set the dependent and independent variables
            y = df[column_names[0]]
            X = df.drop([column_names[0]], axis=1)
            # estimate the logistic regression model
            logit_model = sm.Logit(y, X)
            logit_model_result = logit_model.fit()
             # convert the model summary to a dataframe
            result_df = pd.read_html(logit_model_result.summary().tables[1].as_html(), header=0, index_col=0)[0]
             # pass the result dataframe to the template
            result = pd.read_html(logit_model_result.summary().tables[0].as_html(), header=0, index_col=0)[0]
            context = {
                "result_df":result_df ,
                "object":object,
                "result":result,
                
            }
            return render(request, 'logit_app/logistic_result.html', context)
        elif analysis_type == 'linear':
            # 重回帰分析
            y = df[column_names[0]]
            X = df.drop([column_names[0]], axis=1)
            # 標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # モデル構築
            X = sm.add_constant(X) # 定数項を追加
            model = sm.OLS(y, X)
            model_result = model.fit()

            # 結果の取得
            result_df = pd.read_html(model_result.summary().tables[1].as_html(), header=0, index_col=0)[0]
            result = pd.read_html(model_result.summary().tables[0].as_html(), header=0, index_col=0)[0]

            # 結果をテンプレートに渡す
            context = {
              'result_df': result_df,
              'object':object,
              'result': result
            }
            return render(request, 'logit_app/linear_result.html', context)
    else:
        # render the upload form
        return render(request, 'logit_app/index.html')
