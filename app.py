import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import traceback
import sys

# Print initialization message
print("Starting Regime-Switching Portfolio Optimization App...")

from modules.data_handler import DataHandler
from modules.regime_detection import RegimeDetector
from modules.regime_prediction import RegimePredictor
from modules.portfolio_optimization import PortfolioOptimizer
from modules.performance_metrics import PerformanceAnalyzer
from utils.visualizations import plot_portfolio_performance, plot_regime_transitions, plot_feature_importance
from utils.helpers import load_asset_classes, format_asset_returns_summary

# Print successful import message
print("All modules imported successfully")

# Set page configuration
st.set_page_config(
    page_title="Regime-Switching Portfolio Optimization",
    page_icon="📈",
    layout="wide"
)

# Application title and description
st.title("Regime-Switching Portfolio Optimization")
st.markdown("""
    This application implements a data-driven framework for regime-switching portfolio optimization
    using a combination of machine learning and traditional financial approaches.
""")
st.markdown("""
    The idea of regime-switching portfolio optimization originates from a 
    [Dissertation by P. Pomorski](https://discovery.ucl.ac.uk/id/eprint/10192012/2/Thesis_Piotr_Pomorski_Final.pdf) that 
    Developing and testing a regime-switching-based detection-prediction-optimization framework in financial markets to 
    assist portfolio managers in adapting to changing market conditions.
""")
st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-top: 20px;">
        <p style="font-weight: bold; color: green;">Created by:</p>
        <a href="https://www.linkedin.com/in/danyyudha" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 style="width: 20px; height: 20px;">
        </a>
        <p><b>Dany Yudha Putra Haque</b></p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for configurations
st.sidebar.header("Configuration")

# Data selection section
st.sidebar.subheader("Data Selection")

# Asset class selection
available_asset_classes = load_asset_classes()
selected_assets = st.sidebar.multiselect(
    "Select Asset Classes",
    available_asset_classes,
    default=available_asset_classes[:3]
)

if not selected_assets:
    st.warning("Please select at least one asset class.")
    st.stop()

# Date range selection
today = dt.date.today()
default_start_date = today - relativedelta(years=5)
default_end_date = today

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# Model parameters section
st.sidebar.subheader("Model Parameters")

# Add explanation section
with st.sidebar.expander("📘 Penjelasan Parameter"):
    st.markdown("""
    ### Pengaruh Parameter Model
    
    **Regime Detection:**
    - **KAMA Period:** 
        - **Tinggi (>50):** Lebih sedikit responsif terhadap perubahan harga jangka pendek, menghasilkan sinyal regime yang lebih stabil tetapi lambat.
        - **Rendah (<20):** Lebih responsif terhadap perubahan harga jangka pendek, menghasilkan lebih banyak sinyal (bisa false signal).
    
    - **Number of Markov States:** 
        - **Tinggi (4):** Mendeteksi lebih banyak nuansa dalam kondisi pasar (bull, bear, transisi bull-to-bear, transisi bear-to-bull).
        - **Rendah (2):** Hanya mendeteksi kondisi bull dan bear, lebih sederhana dan kurang sensitif terhadap transisi.
    
    **Regime Prediction:**
    - **Prediction Horizon:** 
        - **Tinggi (>15):** Memprediksi lebih jauh ke depan (akurasi lebih rendah).
        - **Rendah (<5):** Memprediksi hanya dalam jangka pendek (akurasi lebih tinggi).
    
    - **Feature Engineering:** 
        - **Basic:** Menggunakan fitur teknikal dasar, lebih cepat tapi kurang akurat.
        - **Advanced:** Menambahkan indikator teknikal yang lebih kompleks.
        - **Custom:** Menggunakan semua fitur termasuk cross-asset interactions.
    
    **Portfolio Optimization:**
    - **Risk Aversion:** 
        - **Tinggi (>7):** Lebih mengutamakan pengurangan risiko, portofolio lebih terdiversifikasi.
        - **Rendah (<3):** Lebih mengutamakan return, portofolio lebih terkonsentrasi pada aset berisiko tinggi.
    
    - **Transaction Cost:** 
        - **Tinggi (>0.5%):** Mengurangi frekuensi rebalancing, portofolio lebih stabil.
        - **Rendah (<0.1%):** Memungkinkan rebalancing lebih sering, lebih responsif terhadap perubahan pasar.
    
    - **Maximum Allocation:** 
        - **Tinggi (>60%):** Memungkinkan konsentrasi tinggi di satu aset, potensial return lebih tinggi tapi juga risiko lebih tinggi.
        - **Rendah (<20%):** Memaksa diversifikasi yang lebih luas, mengurangi risiko tapi juga potensial return.
    """)

# Regime Detection parameters
kama_period = st.sidebar.slider("KAMA Period", 10, 100, 21, help="Periode untuk Kaufman's Adaptive Moving Average. Nilai tinggi = sinyal lebih stabil tetapi lambat, nilai rendah = lebih responsif tetapi lebih banyak false signal.")
ms_regimes = st.sidebar.slider("Number of Markov States", 2, 4, 2, help="Jumlah regime yang dideteksi. 2 = bull/bear, 4 = bull/bear/transisi bull-to-bear/transisi bear-to-bull.")

# Regime Prediction parameters
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 30, 5, help="Jangka waktu prediksi. Horizon lebih panjang = prediksi lebih jauh ke masa depan (akurasi lebih rendah).")
features_option = st.sidebar.selectbox(
    "Feature Engineering",
    ["Basic", "Advanced", "Custom"],
    help="Kompleksitas fitur. Basic = fitur sederhana & cepat, Advanced = lebih banyak indikator teknis, Custom = semua fitur termasuk interaksi antar aset."
)

# Portfolio Optimization parameters
risk_aversion = st.sidebar.slider("Risk Aversion", 1, 10, 5, help="Tingkat penghindaran risiko. Nilai tinggi = lebih mengutamakan pengurangan risiko, nilai rendah = lebih agresif mencari return tinggi.")
transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1, 0.05, help="Biaya transaksi. Nilai tinggi = rebalancing lebih jarang, nilai rendah = rebalancing lebih sering.")
max_allocation = st.sidebar.slider("Maximum Allocation (%)", 0, 100, 40, 5, help="Alokasi maksimum per aset. Nilai tinggi = konsentrasi pada aset tertentu diperbolehkan, nilai rendah = memaksa diversifikasi lebih luas.")

# Main application logic
try:
    with st.spinner("Loading and processing data..."):
        # Initialize data handler
        data_handler = DataHandler()
        prices_df = data_handler.fetch_data(selected_assets, start_date, end_date)
        returns_df = data_handler.calculate_returns(prices_df)
        
        # Display basic data information
        st.subheader("Asset Data Overview")
        
        with st.expander("📊 Penjelasan Data", expanded=True):
            st.markdown("""
            **Tampilan Data:**
            - **Harga Terakhir**: Menampilkan harga penutupan 5 hari terakhir untuk setiap aset
            - **Ringkasan Return**: Menampilkan statistik penting untuk setiap aset selama periode yang dipilih
            
            **Penjelasan Metrik:**
            - **Annualized Return**: Return tahunan, nilai lebih tinggi = kinerja lebih baik
            - **Annualized Volatility**: Risiko tahunan, nilai lebih rendah = risiko lebih rendah
            - **Sharpe Ratio**: Return per unit risiko, nilai lebih tinggi = efisiensi lebih baik
            - **Max Drawdown**: Penurunan maksimum dari puncak ke lembah, nilai absolut lebih kecil = risiko lebih rendah
            - **Skewness**: Kecondongan distribusi return, nilai positif = potensi return ekstrem positif lebih besar
            - **Kurtosis**: Ekor distribusi, nilai lebih tinggi = lebih banyak kejadian ekstrem (fat tails)
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Harga Terakhir (5 Hari):**")
            st.dataframe(prices_df.tail())
            
        with col2:
            st.write("**Ringkasan Return:**")
            st.dataframe(format_asset_returns_summary(returns_df))
    
    # Regime Detection tab
    tab1, tab2, tab3, tab4 = st.tabs(["Regime Detection", "Regime Prediction", "Portfolio Optimization", "Performance Analysis"])
    
    with tab1:
        st.subheader("Market Regime Detection")
        
        with st.expander("🔍 Penjelasan Regime Detection", expanded=True):
            st.markdown("""
            **Deteksi Regime:**
            
            Sistem ini menggunakan kombinasi Kaufman's Adaptive Moving Average (KAMA) dan Markov-Switching Regression (MSR) untuk mendeteksi perubahan regime pasar.
            
            **Jenis-jenis Regime:**
            - **Bull Market (1):** Regime pasar naik dengan volatilitas rendah
            - **Bear Market (0):** Regime pasar turun dengan volatilitas tinggi
            - **Transisi (2/3):** Periode transisi antara bull dan bear (jika menggunakan 4 states)
            
            **Interpretasi Grafik:**
            - Area berwarna menunjukkan regime yang berbeda
            - Perubahan warna menandakan transisi regime
            - Perhatikan bagaimana harga aset berperilaku di setiap regime
            
            **Regime Statistics:**
            - Menunjukkan karakteristik statistik dari setiap regime
            - Return dan risiko yang berbeda di setiap regime
            """)
        
        with st.spinner("Detecting market regimes..."):
            # Regime detection
            regime_detector = RegimeDetector(kama_period=kama_period, ms_states=ms_regimes)
            regimes_df = regime_detector.detect_regimes(returns_df)
            
            # Plot regimes
            st.pyplot(plot_regime_transitions(prices_df, regimes_df))
            
            # Display regime statistics
            st.subheader("Regime Statistics")
            regime_stats = regime_detector.get_regime_statistics(returns_df, regimes_df)
            st.dataframe(regime_stats)
    
    with tab2:
        st.subheader("Regime Prediction")
        with st.spinner("Training prediction model..."):
            # Regime prediction
            features_df = data_handler.engineer_features(prices_df, returns_df, option=features_option)
            regime_predictor = RegimePredictor(n_estimators=100, horizon=prediction_horizon)
            train_results, test_results = regime_predictor.train_model(features_df, regimes_df)
            
            # Display model performance
            st.subheader("Model Performance")
            st.write(f"Training Accuracy: {train_results['accuracy']:.2f}")
            st.write(f"Testing Accuracy: {test_results['accuracy']:.2f}")
            
            # Display feature importance
            st.subheader("Feature Importance")
            st.pyplot(plot_feature_importance(regime_predictor.get_feature_importance(), features_df.columns))
            
            # Display predictions
            st.subheader("Regime Predictions")
            predictions_df = regime_predictor.predict(features_df)
            st.dataframe(predictions_df.tail(20))
    
    with tab3:
        st.subheader("Portfolio Optimization")
        with st.spinner("Optimizing portfolio..."):
            # Portfolio optimization
            optimizer = PortfolioOptimizer(
                risk_aversion=risk_aversion,
                transaction_cost=transaction_cost/100,  # Convert to decimal
                max_allocation=max_allocation/100      # Convert to decimal
            )
            
            weights_df = optimizer.optimize_portfolio(
                returns_df, 
                regimes_df, 
                predictions_df
            )
            
            # Display portfolio weights
            st.subheader("Portfolio Weights Over Time")
            st.line_chart(weights_df)
            
            # Display latest allocation
            st.subheader("Latest Portfolio Allocation")
            latest_weights = weights_df.iloc[-1].to_dict()
            
            # Create a pie chart of latest weights
            labels = list(latest_weights.keys())
            sizes = list(latest_weights.values())
            
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Performance Analysis")
        with st.spinner("Calculating performance metrics..."):
            # Performance analysis
            analyzer = PerformanceAnalyzer()
            
            # Calculate portfolio returns
            portfolio_returns = analyzer.calculate_portfolio_returns(returns_df, weights_df)
            
            # Calculate benchmark returns (using equal weight)
            benchmark_weights = pd.DataFrame(
                np.ones((len(returns_df), len(selected_assets))) / len(selected_assets),
                index=returns_df.index,
                columns=returns_df.columns
            )
            benchmark_returns = analyzer.calculate_portfolio_returns(returns_df, benchmark_weights)
            
            # Calculate performance metrics
            metrics = analyzer.calculate_performance_metrics(portfolio_returns, benchmark_returns)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strategy Performance Metrics")
                metrics_df = pd.DataFrame([metrics["portfolio"]])
                st.dataframe(metrics_df)
                
            with col2:
                st.subheader("Benchmark Performance Metrics")
                benchmark_df = pd.DataFrame([metrics["benchmark"]])
                st.dataframe(benchmark_df)
            
            # Plot cumulative returns
            st.subheader("Cumulative Performance")
            st.pyplot(plot_portfolio_performance(portfolio_returns, benchmark_returns))
            
            # Drawdown analysis
            st.subheader("Drawdown Analysis")
            portfolio_dd = analyzer.calculate_drawdown(portfolio_returns)
            benchmark_dd = analyzer.calculate_drawdown(benchmark_returns)
            
            dd_df = pd.DataFrame({
                'Strategy Drawdown': portfolio_dd,
                'Benchmark Drawdown': benchmark_dd
            })
            
            st.line_chart(dd_df)

except Exception as e:
    error_msg = f"An error occurred: {str(e)}"
    tb = traceback.format_exc()
    print(error_msg)
    print(tb)
    
    st.error(error_msg)
    st.write("Please check your inputs and try again.")
    
    # Show detailed error information for debugging
    with st.expander("Error Details"):
        st.code(tb)
