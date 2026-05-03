# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:06:24 2026

@author: kerem
"""

import streamlit as st
import plotly.graph_objects as go
from hausdorff import get_boundary_points, get_hausdorff_details

st.set_page_config(page_title="Hausdorff Mesafe Laboratuvarı", layout="wide")

st.title("Hausdorff Mesafe Laboratuvarı")
st.markdown("Kompakt kümeler arasındaki metrik uzaklığı interaktif olarak hesaplayın ve görselleştirin.")

with st.sidebar:
    st.header("Parametreler")
    
    st.subheader("Kümeleri Tanımla")
    eq_a = st.text_input("Küme A Denklemi:", "x**2 + y**2 = 4")
    eq_b = st.text_input("Küme B Denklemi:", "abs(x) + abs(y) = 1")
    
    st.subheader("Hesaplama Ayarları")
    res = st.slider("Hassasiyet (Resolution)", min_value=50, max_value=500, value=250, step=50,
                    help="Değer arttıkça şekil sınırları daha hassas bulunur ancak hesaplama süresi uzar.")
    
    st.write("**Arama Sınırları (Grid Bounds)**")
    col1, col2 = st.columns(2)
    with col1:
        grid_min = st.number_input("Min", value=-6.0, step=1.0)
    with col2:
        grid_max = st.number_input("Max", value=6.0, step=1.0)

if st.button("Analiz Et ve Çiz", use_container_width=True):
    with st.spinner('Matematiksel sınırlar hesaplanıyor...'):
        bounds = (grid_min, grid_max)
        A = get_boundary_points(eq_a, resolution=res, grid_bounds=bounds)
        B = get_boundary_points(eq_b, resolution=res, grid_bounds=bounds)
        
        if A.size > 0 and B.size > 0:
            h, p1, p2, yon = get_hausdorff_details(A, B)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=A[:,0], y=A[:,1], mode='markers', 
                                     name='Küme A', marker=dict(color='royalblue', size=3)))
            fig.add_trace(go.Scatter(x=B[:,0], y=B[:,1], mode='markers', 
                                     name='Küme B', marker=dict(color='firebrick', size=3)))
            
            fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], 
                                     mode='lines+markers', name=f'Hausdorff: {h:.3f}',
                                     line=dict(color='gold', width=4, dash='dash'),
                                     marker=dict(size=8, symbol='x')))
            
            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=1),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black'),
                yaxis_title="Y Ekseni",
                xaxis_title="X Ekseni",
                title="Yönlü Uzaklık Vektörü Analizi"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"### Hausdorff Mesafesi (h): **{h:.4f}** birim")
            st.info(f"**Analiz:** Bu mesafe, **{yon}** yönündeki uzaklığın maksimum olduğu noktadan kaynaklanıyor. Yukarıdaki grafikte sarı kesik çizgi ile gösterilen vektör, bu kritik mesafeyi temsil eder.")
            
        else:
            st.error("Hata: Denklemlerden biri veya her ikisi için sınır noktaları bulunamadı. Lütfen denklemleri veya 'Arama Sınırları' (Grid Bounds) değerlerini kontrol edin.")