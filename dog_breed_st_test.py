# streamlit run <filename>

import os
from dog_breed_xgb import # ???
#%%
try:
    import streamlit as st 
except ImportError:
    os.system('pip install streamlit')
    import streamlit as st
    
st.write('Hello World!')

test = st.text_input('Enter your name:')
button = st.button('Submit', key=1)

if button:
    st.write(f'Hello, {test}!')