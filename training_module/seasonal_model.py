"""
Seasonal and temporal stress prediction model using Facebook's Prophet.

This module trains a time-series forecasting model to identify seasonal trends
in student stress levels, capturing patterns like exam season spikes and semester
stress variations.

Classes:
    SeasonalModel: Wrapper for Prophet-based temporal stress analysis

Functions:
    train_seasonal_from_logs: Train Prophet model from daily logs DataFrame
    forecast_stress_trend: Generate stress forecasts for upcoming weeks
"""

import pandas as pd
from prophet import Prophet
from typing import Dict, Any, Tuple, Optional


def train_seasonal_from_logs(logs_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Train a Prophet model on historical daily logs for seasonal stress analysis.
    
    Args:
        logs_df (pd.DataFrame): DataFrame with columns ['log_date', 'predicted_stress_level'].
                               Must have at least 30 observations for meaningful results.
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'model': Trained Prophet model instance
            - 'trend': Extracted trend component (low/medium/high)
            - 'seasonality': Detected seasonal pattern strength (0-100%)
            - 'forecast': 30-day ahead forecast DataFrame
    
    Raises:
        ValueError: If required columns are missing or insufficient data rows
    
    Example:
        >>> logs = pd.DataFrame({'log_date': pd.date_range('2023-01-01', periods=60),
        ...                       'predicted_stress_level': [2.5]*60})
        >>> result = train_seasonal_from_logs(logs)
        >>> print(result['trend'])
        'stable'
    """
    if logs_df.empty or logs_df.shape[0] < 7:
        return {
            'model': None,
            'trend': 'insufficient_data',
            'seasonality': 0.0,
            'forecast': None,
            'avg_stress': 0.0,
        }
    
    # Prepare data for Prophet (requires 'ds' and 'y' column names)
    prophet_df = logs_df[['log_date', 'predicted_stress_level']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df = prophet_df.sort_values('ds').drop_duplicates(subset=['ds'], keep='last')
    
    if prophet_df.shape[0] < 7:
        return {
            'model': None,
            'trend': 'insufficient_data',
            'seasonality': 0.0,
            'forecast': None,
            'avg_stress': float(prophet_df['y'].mean()) if not prophet_df.empty else 0.0,
        }
    
    try:
        # Train Prophet model with yearly and weekly seasonality
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.95,
        )
        model.fit(prophet_df)
        
        # Generate 30-day forecast
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # Extract trend from last 14 days
        recent_trend = prophet_df['y'].tail(14).mean()
        overall_avg = prophet_df['y'].mean()
        
        if recent_trend > overall_avg + 0.5:
            trend = 'increasing'
        elif recent_trend < overall_avg - 0.5:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        # Calculate seasonality strength (variance explained by seasonality)
        seasonality_strength = _calculate_seasonality_strength(prophet_df, forecast)
        
        return {
            'model': model,
            'trend': trend,
            'seasonality': seasonality_strength,
            'forecast': forecast,
            'avg_stress': float(overall_avg),
        }
    except Exception as e:
        return {
            'model': None,
            'trend': f'error: {str(e)}',
            'seasonality': 0.0,
            'forecast': None,
            'avg_stress': float(prophet_df['y'].mean()) if not prophet_df.empty else 0.0,
        }


def forecast_stress_trend(model_bundle: Dict[str, Any], weeks: int = 4) -> Optional[pd.DataFrame]:
    """
    Generate stress forecast for upcoming weeks using trained seasonal model.
    
    Args:
        model_bundle (Dict): Output from train_seasonal_from_logs()
        weeks (int): Number of weeks to forecast (default: 4)
    
    Returns:
        Optional[pd.DataFrame]: Forecast DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
                               or None if model unavailable
    
    Example:
        >>> forecast_df = forecast_stress_trend(model_bundle, weeks=2)
        >>> forecast_df[['ds', 'yhat']].tail()
    """
    if not model_bundle or model_bundle.get('model') is None:
        return None
    
    try:
        model = model_bundle['model']
        future = model.make_future_dataframe(periods=weeks * 7)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(weeks * 7)
    except Exception:
        return None


def get_seasonal_insight(model_bundle: Dict[str, Any]) -> str:
    """
    Generate human-readable insight about stress seasonality.
    
    Args:
        model_bundle (Dict): Output from train_seasonal_from_logs()
    
    Returns:
        str: Seasonal insight message
    
    Example:
        >>> insight = get_seasonal_insight(model_bundle)
        >>> print(insight)
        'Stress is currently increasing with 35% seasonal variation. Watch for exam season spikes.'
    """
    trend = model_bundle.get('trend', 'unknown')
    seasonality = model_bundle.get('seasonality', 0.0)
    
    trend_msg = {
        'increasing': 'Stress levels are increasing. Consider intensifying self-care habits.',
        'decreasing': 'Great! Stress levels are trending downward. Maintain current habits.',
        'stable': 'Stress levels are stable. Continue current wellness routine.',
        'insufficient_data': 'Need more daily logs to detect trends (minimum 7 days).',
        'unknown': 'Unable to determine stress trend.',
    }.get(trend, 'Unable to determine stress trend.')
    
    if isinstance(seasonality, (int, float)) and seasonality > 30:
        season_msg = f'Strong seasonal pattern detected ({seasonality:.0f}% variation). Stress may spike during exam periods.'
    elif isinstance(seasonality, (int, float)) and seasonality > 15:
        season_msg = f'Moderate seasonal variation detected ({seasonality:.0f}%). Watch for stress increases.'
    else:
        season_msg = 'No strong seasonal pattern detected yet.'
    
    return f"{trend_msg} {season_msg}"


def _calculate_seasonality_strength(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> float:
    """
    Calculate seasonality strength as percentage variance from trend.
    
    Args:
        actual_df (pd.DataFrame): Historical data with 'y' column
        forecast_df (pd.DataFrame): Prophet forecast with 'trend' column
    
    Returns:
        float: Seasonality strength (0-100%)
    """
    try:
        if 'trend' not in forecast_df.columns or forecast_df.empty:
            return 0.0
        
        # Match dates and calculate residuals
        actual_df_copy = actual_df.copy()
        actual_df_copy.columns = ['ds', 'y']
        
        merged = actual_df_copy.merge(
            forecast_df[['ds', 'trend']], 
            on='ds', 
            how='inner'
        )
        
        if merged.empty:
            return 0.0
        
        # Seasonality = (variance of residuals) / (variance of y) * 100
        variance_y = merged['y'].var()
        variance_residual = (merged['y'] - merged['trend']).var()
        
        if variance_y == 0 or variance_y < 1e-6:
            return 0.0
        
        seasonality = min((variance_residual / variance_y) * 100, 100.0)
        return float(seasonality)
    except Exception:
        return 0.0
