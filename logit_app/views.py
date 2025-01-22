from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from django.core.exceptions import ValidationError
import logging
from io import StringIO

logger = logging.getLogger(__name__)

def validate_dataframe(df, analysis_type):
    """Validates the structure and content of the DataFrame for analysis."""
    if df.empty:
        raise ValidationError("Uploaded file is empty.")
    if len(df.columns) < 2:
        raise ValidationError("CSV must have at least two columns (a target variable and at least one predictor).")

    if analysis_type == 'logistic':
        target_series = df.iloc[:, 0]
        if not all(x in [0, 1] for x in target_series.dropna()):
            raise ValidationError("Logistic regression requires the target variable to be binary (0 or 1).")
    elif analysis_type == 'linear':
        try:
            df.iloc[:,0].astype(float)
        except ValueError:
                raise ValidationError("Linear regression requires the target variable to be numeric.")

def index(request):
    if request.method == 'POST':
        analysis_type = request.POST.get('analysis_type')
        try:
            csv_file = request.FILES['csv_file']
            try:
                df = pd.read_csv(csv_file, encoding="utf-8")
            except UnicodeDecodeError as e:
                logger.error(f'UnicodeDecodeError during CSV read: {e}')
                df = pd.read_csv(csv_file, encoding="cp932")  # Try Shift_JIS as a fallback
            except Exception as e:
                 logger.error(f'Failed to read CSV file: {e}')
                 return render(request, 'logit_app/index.html', {'error': f'Failed to read CSV file: {e}'})

            try:
              validate_dataframe(df, analysis_type)
            except ValidationError as e:
                 logger.error(f'Validation Error: {e}')
                 return render(request, 'logit_app/index.html', {'error': str(e)})

            column_names = list(df)
            object = column_names[0]
            if analysis_type == 'logistic':
                 # set the dependent and independent variables
                y = df[column_names[0]]
                X = df.drop([column_names[0]], axis=1)
                # estimate the logistic regression model
                logit_model = sm.Logit(y, X)
                try:
                   logit_model_result = logit_model.fit()
                except Exception as e:
                     logger.error(f'Error during logistic regression model fitting: {e}')
                     return render(request, 'logit_app/index.html', {'error': f'Error during logistic regression model fitting: {e}'})
                # convert the model summary to a dataframe
                try:
                   result_df = pd.read_html(StringIO(logit_model_result.summary().tables[1].as_html()), header=0, index_col=0)[0]
                except Exception as e:
                    logger.error(f'Error extracting logistic regression table: {e}')
                    return render(request, 'logit_app/index.html', {'error': f'Error extracting logistic regression table: {e}'})
                 # pass the result dataframe to the template
                try:
                    result = pd.read_html(StringIO(logit_model_result.summary().tables[0].as_html()), header=0, index_col=0)[0]
                except Exception as e:
                   logger.error(f'Error extracting logistic regression table: {e}')
                   return render(request, 'logit_app/index.html', {'error': f'Error extracting logistic regression table: {e}'})
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
                try:
                    model_result = model.fit()
                except Exception as e:
                     logger.error(f'Error during linear regression model fitting: {e}')
                     return render(request, 'logit_app/index.html', {'error': f'Error during linear regression model fitting: {e}'})

                # 結果の取得
                try:
                    result_df = pd.read_html(StringIO(model_result.summary().tables[1].as_html()), header=0, index_col=0)[0]
                except Exception as e:
                    logger.error(f'Error extracting linear regression table: {e}')
                    return render(request, 'logit_app/index.html', {'error': f'Error extracting linear regression table: {e}'})
                try:
                   result = pd.read_html(StringIO(model_result.summary().tables[0].as_html()), header=0, index_col=0)[0]
                except Exception as e:
                  logger.error(f'Error extracting linear regression table: {e}')
                  return render(request, 'logit_app/index.html', {'error': f'Error extracting linear regression table: {e}'})

                # 結果をテンプレートに渡す
                context = {
                  'result_df': result_df,
                  'object':object,
                  'result': result
                }
                return render(request, 'logit_app/linear_result.html', context)
        except Exception as e:
              logger.error(f'An error occurred during upload or analysis: {e}')
              return render(request, 'logit_app/index.html', {'error': f'An error occurred during upload or analysis: {e}'})
    else:
        # render the upload form
        return render(request, 'logit_app/index.html')
