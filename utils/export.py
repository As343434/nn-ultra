"""Export helpers: pickle, code, torch, CSV."""
import io
import pickle
import streamlit as st


def download_pickle(label: str, obj, filename: str):
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    st.download_button(label, buf.getvalue(), filename, "application/octet-stream")


def download_code(label: str, code: str, filename: str):
    st.download_button(label, code, filename, "text/plain")


def download_torch(label: str, model, filename: str):
    try:
        import torch, io as _io
        buf = _io.BytesIO()
        torch.save(model.state_dict(), buf)
        st.download_button(label, buf.getvalue(), filename, "application/octet-stream")
    except Exception as e:
        st.warning(f"Torch export: {e}")


def download_csv(label: str, df, filename: str):
    st.download_button(label, df.to_csv(index=False), filename, "text/csv")
