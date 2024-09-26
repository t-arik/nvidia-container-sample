FROM pytorch/pytorch

COPY simple.py .

CMD ["python", "simple.py"]
