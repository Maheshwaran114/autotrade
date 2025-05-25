#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Analysis and Reporting Module.

This module provides comprehensive tools for analyzing backtest results 
and generating performance reports for trading strategies. It includes
functionality for calculating performance metrics, visualizing results,
and generating detailed reports for both ML-enhanced and traditional strategies.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """
    Class for analyzing backtest results and generating comprehensive reports.
    
    This analyzer calculates performance metrics, creates visualizations,
    and generates detailed reports on strategy performance. It provides
    special handling for ML-enhanced strategies to compare their performance
    with traditional approaches.
    """
    
    def __init__(
        self,
        results_dir: str = "reports/backtest",
        benchmark_data: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the backtest analyzer.
        
        Args:
            results_dir: Directory to save analysis results
            benchmark_data: DataFrame with benchmark return data (e.g., index returns)
        """
        self.results_dir = results_dir
        self.benchmark_data = benchmark_data
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize results container
        self.strategies = {}
        
    def add_strategy_result(
        self,
        strategy_name: str,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
        metadata: Dict = None,
    ) -> None:
        """
        Add backtest results for a trading strategy.
        
        Args:
            strategy_name: Name of the strategy
            trades: DataFrame with trade results
            equity_curve: DataFrame with equity curve
            metadata: Dictionary with additional metadata
        """
        logger.info(f"Adding backtest results for strategy: {strategy_name}")
        
        # Store results
        self.strategies[strategy_name] = {
            'trades': trades,
            'equity_curve': equity_curve,
            'metadata': metadata or {},
            'metrics': None,  # Will be populated by calculate_metrics
        }
    
    def calculate_metrics(self, strategy_name: str) -> Dict:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with performance metrics
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        strategy_data = self.strategies[strategy_name]
        trades = strategy_data['trades']
        equity = strategy_data['equity_curve']
        
        metrics = {}
        
        # Basic performance metrics
        if 'pnl' in trades.columns:
            metrics['total_return'] = trades['pnl'].sum()
            metrics['win_rate'] = len(trades[trades['pnl'] > 0]) / len(trades) if len(trades) > 0 else 0
            metrics['profit_factor'] = (
                abs(trades[trades['pnl'] > 0]['pnl'].sum()) / 
                abs(trades[trades['pnl'] < 0]['pnl'].sum())
                if abs(trades[trades['pnl'] < 0]['pnl'].sum()) > 0 else float('inf')
            )
            
        # Risk metrics
        if 'equity' in equity.columns:
            returns = equity['equity'].pct_change().dropna()
            metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            metrics['max_drawdown'] = self._calculate_max_drawdown(equity['equity'])
            metrics['calmar_ratio'] = (returns.mean() * 252) / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Trade statistics
        metrics['num_trades'] = len(trades)
        metrics['avg_trade_duration'] = self._calculate_avg_trade_duration(trades)
        
        # ML-specific metrics if available
        if strategy_data['metadata'].get('ml_enhanced', False):
            metrics['ml_metrics'] = self._calculate_ml_metrics(strategy_name)
        
        # Store metrics
        self.strategies[strategy_name]['metrics'] = metrics
        
        return metrics
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_series: Series with equity curve
            
        Returns:
            Maximum drawdown as a percentage
        """
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        return drawdown.min()
    
    def _calculate_avg_trade_duration(self, trades: pd.DataFrame) -> float:
        """
        Calculate average trade duration.
        
        Args:
            trades: DataFrame with trade results
            
        Returns:
            Average trade duration in days
        """
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            durations = (trades['exit_time'] - trades['entry_time']).dt.total_seconds() / (60 * 60 * 24)
            return durations.mean()
        return 0
    
    def _calculate_ml_metrics(self, strategy_name: str) -> Dict:
        """
        Calculate ML-specific metrics.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with ML-specific metrics
        """
        strategy_data = self.strategies[strategy_name]
        trades = strategy_data['trades']
        ml_metrics = {}
        
        # Check if ML-specific columns are available
        if 'day_type' in trades.columns:
            # Performance by day type
            ml_metrics['performance_by_day_type'] = trades.groupby('day_type')['pnl'].agg([
                'count', 'mean', 'sum', 'std'
            ]).reset_index()
            
        if 'signal_confidence' in trades.columns:
            # Performance by signal confidence
            confidence_bins = pd.cut(trades['signal_confidence'], bins=5)
            ml_metrics['performance_by_confidence'] = trades.groupby(confidence_bins)['pnl'].agg([
                'count', 'mean', 'sum', 'std'
            ]).reset_index()
        
        return ml_metrics
    
    def compare_strategies(self, strategy_names: List[str] = None) -> pd.DataFrame:
        """
        Compare performance metrics across strategies.
        
        Args:
            strategy_names: List of strategy names to compare (if None, compare all)
            
        Returns:
            DataFrame with performance comparison
        """
        if strategy_names is None:
            strategy_names = list(self.strategies.keys())
            
        # Calculate metrics for all strategies if not already done
        for strategy_name in strategy_names:
            if self.strategies[strategy_name]['metrics'] is None:
                self.calculate_metrics(strategy_name)
                
        # Extract metrics for each strategy
        comparison = {}
        for strategy_name in strategy_names:
            metrics = self.strategies[strategy_name]['metrics']
            for metric, value in metrics.items():
                if metric != 'ml_metrics':  # Skip ML-specific metrics
                    if metric not in comparison:
                        comparison[metric] = {}
                    comparison[metric][strategy_name] = value
        
        # Convert to DataFrame
        return pd.DataFrame(comparison)
    
    def generate_report(
        self,
        strategy_name: str,
        output_file: str = None,
        include_plots: bool = True
    ) -> str:
        """
        Generate a comprehensive performance report for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            output_file: Path to save the report (if None, use default)
            include_plots: Whether to include plots in the report
            
        Returns:
            Path to the generated report
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
            
        # Calculate metrics if not already done
        if self.strategies[strategy_name]['metrics'] is None:
            self.calculate_metrics(strategy_name)
            
        # Default output file
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                self.results_dir, 
                f"{strategy_name}_report_{timestamp}.html"
            )
            
        strategy_data = self.strategies[strategy_name]
        metrics = strategy_data['metrics']
        
        # Generate plots if requested
        if include_plots:
            plots_dir = os.path.join(self.results_dir, 'plots', strategy_name)
            os.makedirs(plots_dir, exist_ok=True)
            self._generate_plots(strategy_name, plots_dir)
        
        # Generate HTML report
        report_content = self._generate_html_report(strategy_name, metrics, plots_dir if include_plots else None)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Report generated for {strategy_name} at {output_file}")
        return output_file
    
    def _generate_plots(self, strategy_name: str, plots_dir: str) -> Dict[str, str]:
        """
        Generate performance plots for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        strategy_data = self.strategies[strategy_name]
        trades = strategy_data['trades']
        equity = strategy_data['equity_curve']
        
        plot_files = {}
        
        # Equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity['equity'])
        plt.title(f"{strategy_name} - Equity Curve")
        plt.tight_layout()
        equity_plot = os.path.join(plots_dir, "equity_curve.png")
        plt.savefig(equity_plot)
        plt.close()
        plot_files['equity_curve'] = equity_plot
        
        # Drawdown chart
        plt.figure(figsize=(12, 6))
        cummax = equity['equity'].cummax()
        drawdown = (equity['equity'] - cummax) / cummax
        plt.fill_between(equity.index, drawdown, 0, alpha=0.3, color='r')
        plt.title(f"{strategy_name} - Drawdown")
        plt.tight_layout()
        drawdown_plot = os.path.join(plots_dir, "drawdown.png")
        plt.savefig(drawdown_plot)
        plt.close()
        plot_files['drawdown'] = drawdown_plot
        
        # Trade PnL histogram
        if 'pnl' in trades.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(trades['pnl'], bins=50)
            plt.title(f"{strategy_name} - Trade PnL Distribution")
            plt.tight_layout()
            pnl_plot = os.path.join(plots_dir, "pnl_distribution.png")
            plt.savefig(pnl_plot)
            plt.close()
            plot_files['pnl_distribution'] = pnl_plot
        
        # ML-specific plots
        if strategy_data['metadata'].get('ml_enhanced', False):
            # Performance by day type
            if 'day_type' in trades.columns:
                plt.figure(figsize=(10, 6))
                day_type_perf = trades.groupby('day_type')['pnl'].sum()
                day_type_perf.plot(kind='bar')
                plt.title(f"{strategy_name} - Performance by Day Type")
                plt.tight_layout()
                day_type_plot = os.path.join(plots_dir, "performance_by_day_type.png")
                plt.savefig(day_type_plot)
                plt.close()
                plot_files['performance_by_day_type'] = day_type_plot
            
            # Performance by signal confidence
            if 'signal_confidence' in trades.columns:
                plt.figure(figsize=(10, 6))
                sns.regplot(x='signal_confidence', y='pnl', data=trades)
                plt.title(f"{strategy_name} - PnL vs Signal Confidence")
                plt.tight_layout()
                confidence_plot = os.path.join(plots_dir, "performance_by_confidence.png")
                plt.savefig(confidence_plot)
                plt.close()
                plot_files['performance_by_confidence'] = confidence_plot
        
        return plot_files
    
    def _generate_html_report(
        self,
        strategy_name: str,
        metrics: Dict,
        plots_dir: Optional[str]
    ) -> str:
        """
        Generate HTML report content.
        
        Args:
            strategy_name: Name of the strategy
            metrics: Dictionary with performance metrics
            plots_dir: Directory with plots (or None if not available)
            
        Returns:
            HTML report content
        """
        strategy_data = self.strategies[strategy_name]
        
        # Start building HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} - Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .metric-value {{ font-weight: bold; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>{strategy_name} - Performance Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add key metrics to table
        for metric in ['total_return', 'win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio', 'num_trades']:
            if metric in metrics:
                value = metrics[metric]
                # Format value based on metric type
                if metric in ['win_rate']:
                    formatted_value = f"{value:.2%}"
                elif metric in ['max_drawdown']:
                    formatted_value = f"{value:.2%}"
                    css_class = 'bad' if value < -0.1 else ''
                elif metric in ['sharpe_ratio', 'calmar_ratio']:
                    formatted_value = f"{value:.3f}"
                    css_class = 'good' if value > 1 else ('bad' if value < 0 else '')
                elif metric in ['profit_factor']:
                    formatted_value = f"{value:.3f}"
                    css_class = 'good' if value > 1 else 'bad'
                elif metric in ['total_return']:
                    formatted_value = f"{value:.2f}"
                    css_class = 'good' if value > 0 else 'bad'
                else:
                    formatted_value = f"{value}"
                    css_class = ''
                
                html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td class="metric-value {css_class}">{formatted_value}</td>
                </tr>
                """
        
        html += """
            </table>
            
            <h2>Trade Statistics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add trade statistics
        if 'avg_trade_duration' in metrics:
            html += f"""
            <tr>
                <td>Average Trade Duration</td>
                <td>{metrics['avg_trade_duration']:.2f} days</td>
            </tr>
            """
        
        # Add ML-specific metrics if available
        if 'ml_metrics' in metrics and metrics['ml_metrics']:
            ml_metrics = metrics['ml_metrics']
            
            html += """
            </table>
            
            <h2>ML Performance Analysis</h2>
            """
            
            if 'performance_by_day_type' in ml_metrics:
                html += """
                <h3>Performance by Day Type</h3>
                <table>
                    <tr>
                        <th>Day Type</th>
                        <th>Trade Count</th>
                        <th>Avg PnL</th>
                        <th>Total PnL</th>
                    </tr>
                """
                
                for _, row in ml_metrics['performance_by_day_type'].iterrows():
                    html += f"""
                    <tr>
                        <td>{row['day_type']}</td>
                        <td>{row['count']}</td>
                        <td class="{'good' if row['mean'] > 0 else 'bad'}">{row['mean']:.2f}</td>
                        <td class="{'good' if row['sum'] > 0 else 'bad'}">{row['sum']:.2f}</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
        
        # Add plots if available
        if plots_dir:
            html += """
            <h2>Performance Charts</h2>
            """
            
            for plot_name, plot_file in self._get_plot_files(plots_dir).items():
                if os.path.exists(plot_file):
                    plot_title = plot_name.replace('_', ' ').title()
                    html += f"""
                    <div class="plot">
                        <h3>{plot_title}</h3>
                        <img src="{os.path.relpath(plot_file, os.path.dirname(plots_dir))}" alt="{plot_title}">
                    </div>
                    """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _get_plot_files(self, plots_dir: str) -> Dict[str, str]:
        """
        Get paths to plot files in the plots directory.
        
        Args:
            plots_dir: Directory with plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_files = {}
        for filename in os.listdir(plots_dir):
            if filename.endswith('.png'):
                name = os.path.splitext(filename)[0]
                plot_files[name] = os.path.join(plots_dir, filename)
        return plot_files


# Simple command-line interface for testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    import random
    
    # Sample equity curve
    dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
    equity = 100000 * (1 + np.cumsum(np.random.normal(0.001, 0.01, size=100)))
    equity_df = pd.DataFrame({
        'equity': equity
    }, index=dates)
    
    # Sample trades
    num_trades = 50
    entry_dates = np.random.choice(dates[:-1], size=num_trades, replace=False)
    trade_durations = np.random.randint(1, 10, size=num_trades)
    exit_dates = [dates[np.where(dates == entry_date)[0][0] + duration] 
                 for entry_date, duration in zip(entry_dates, trade_durations)]
    
    day_types = np.random.choice(['Trend', 'RangeBound', 'Event', 'MildBias', 'Momentum'], size=num_trades)
    signal_confidences = np.random.uniform(0.5, 1.0, size=num_trades)
    returns = np.random.normal(0.002, 0.01, size=num_trades)
    pnls = 100000 * returns
    
    trades_df = pd.DataFrame({
        'entry_time': entry_dates,
        'exit_time': exit_dates,
        'entry_price': np.random.uniform(45000, 50000, size=num_trades),
        'exit_price': np.random.uniform(45000, 50000, size=num_trades),
        'pnl': pnls,
        'day_type': day_types,
        'signal_confidence': signal_confidences
    })
    
    # Create analyzer and add strategy
    analyzer = BacktestAnalyzer(results_dir="reports/backtest_test")
    analyzer.add_strategy_result(
        "test_strategy",
        trades_df,
        equity_df,
        metadata={'ml_enhanced': True}
    )
    
    # Calculate metrics and generate report
    metrics = analyzer.calculate_metrics("test_strategy")
    report_path = analyzer.generate_report("test_strategy")
    
    print(f"Metrics: {metrics}")
    print(f"Report generated at: {report_path}")
