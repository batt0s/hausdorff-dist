# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:06:24 2026

@author: kerem
"""

import streamlit as st
import plotly.graph_objects as go
from hausdorff import get_shape_points, get_hausdorff_details

def get_metric_path(p1, p2, metric):
    """Returns waypoints for drawing the distance path under the given metric."""
    if metric == "cityblock":  # Manhattan: L-shaped path
        # Go horizontal first, then vertical
        mid = [p2[0], p1[1]]
        return [p1[0], mid[0], p2[0]], [p1[1], mid[1], p2[1]]
    elif metric == "chebyshev":  # Chebyshev: diagonal then straight
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        diag = min(abs(dx), abs(dy))
        sx, sy = np.sign(dx), np.sign(dy)
        mid = [p1[0] + sx * diag, p1[1] + sy * diag]
        return [p1[0], mid[0], p2[0]], [p1[1], mid[1], p2[1]]
    else:  # Euclidean, Minkowski, etc.: straight line
        return [p1[0], p2[0]], [p1[1], p2[1]]

st.set_page_config(page_title="Hausdorff Mesafe Laboratuvarı", layout="wide")

st.title("Hausdorff Mesafe Laboratuvarı")
st.markdown("Kompakt kümeler arasındaki metrik uzaklığı interaktif olarak hesaplayın ve görselleştirin.")

with st.sidebar:
    st.header("Parametreler")
    
    st.subheader("Kümeleri Tanımla")
    eq_a = st.text_input("Küme A Denklemi:", "x**2 + y**2 = 4")
    eq_b = st.text_input("Küme B Denklemi:", "abs(x) + abs(y) = 1")

    st.subheader("Metrik Uzay Ayarları")
    metric_type = st.selectbox(
        "Mesafe Metriğini Seçin",
        options=["euclidean", "cityblock", "chebyshev"],
        format_func=lambda x: {
            "euclidean": "Öklid (Kuş Uçuşu / L2)",
            "cityblock": "Manhattan (L1) - |x| + |y|",
            "chebyshev": "Chebyshev (Satranç Tahtası / L∞)",
        }[x]
    )

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
        A = get_shape_points(eq_a, resolution=res, grid_bounds=bounds)
        B = get_shape_points(eq_b, resolution=res, grid_bounds=bounds)
        
        if A.size > 0 and B.size > 0:
            h, d_AB, pA_max, pB_near, d_BA, pB_max, pA_near = get_hausdorff_details(A, B, metric_type=metric_type)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=A[:,0], y=A[:,1], mode='markers', 
                                     name='Küme A', marker=dict(color='royalblue', size=3)))
            fig.add_trace(go.Scatter(x=B[:,0], y=B[:,1], mode='markers', 
                                     name='Küme B', marker=dict(color='firebrick', size=3)))
            
            is_h_AB = (h == d_AB)
            xs, ys = get_metric_path(pA_max, pB_near, metric_type)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, 
                mode='lines+markers', name=f'd(A,B): {d_AB:.3f}',
                line=dict(color='gold' if is_h_AB else 'mediumseagreen', width=4, dash='solid' if is_h_AB else 'dot'),
                marker=dict(size=8, symbol='circle')
            ))

            is_h_BA = (h == d_BA)
            xs, ys = get_metric_path(pB_max, pA_near, metric_type)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, 
                mode='lines+markers', name=f'd(B,A): {d_BA:.3f}',
                line=dict(color='gold' if is_h_BA else 'mediumseagreen', width=4, dash='solid' if is_h_BA else 'dot'),
                marker=dict(size=8, symbol='square')
            ))
            
            fig.update_layout(
                yaxis=dict(scaleanchor="x", scaleratio=1),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=True, zerolinecolor='black'),
                yaxis_title="Y Ekseni",
                xaxis_title="X Ekseni",
                title="Yönlü Uzaklık Vektörü Analizi"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"### Hausdorff Mesafesi: h = **{h:.4f}** birim")
            # st.info(f"**Analiz:** Bu mesafe, **{yon}** yönündeki uzaklığın maksimum olduğu noktadan kaynaklanıyor. Yukarıdaki grafikte sarı kesik çizgi ile gösterilen vektör, bu kritik mesafeyi temsil eder.")
            
        else:
            st.error("Hata: Denklemlerden biri veya her ikisi için sınır noktaları bulunamadı. Lütfen denklemleri veya 'Arama Sınırları' (Grid Bounds) değerlerini kontrol edin.")
