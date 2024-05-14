import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def gaussian_kernel(distance, bandwidth):
    return np.exp(-0.5 * (distance / bandwidth)**2)

def lowess(x, y, bandwidth=0.1, iterations=3):
    n = len(x)
    y_smoothed = np.zeros(n)
    
    for i in range(n):
        weights = gaussian_kernel(np.abs(x - x[i]), bandwidth)
        weights *= np.sqrt(weights)  # "bisquare" weighting
        weights /= np.sum(weights)
        for _ in range(iterations):
            numerator = np.sum(weights * y)
            denominator = np.sum(weights)
            y_smoothed[i] = numerator / denominator
    
    return y_smoothed

def main():
    st.title('Locally Weighted Regression (LOWESS)')
    
    st.sidebar.header('Settings')
    bandwidth = st.sidebar.slider('Bandwidth', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    iterations = st.sidebar.slider('Iterations', min_value=1, max_value=10, value=3, step=1)
    
    st.write('Please provide your data points:')
    data = st.text_area('Input data (x,y)', '1,2\n2,3\n3,4\n4,5\n5,6')
    
    try:
        data_points = np.array([[float(point.split(',')[0]), float(point.split(',')[1])] for point in data.split('\n')])
        
        x = data_points[:, 0]
        y = data_points[:, 1]
        
        y_smoothed = lowess(x, y, bandwidth=bandwidth, iterations=iterations)
        
        fig, ax = plt.subplots()
        ax.scatter(x, y, label='Data Points')
        ax.plot(x, y_smoothed, color='red', label='Smoothed Curve')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        st.pyplot(fig)
    except:
        st.error('Please provide valid data points in the format "x,y" separated by newlines.')

if __name__ == '__main__':
    main()
