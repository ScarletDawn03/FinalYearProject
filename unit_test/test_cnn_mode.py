import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from CNN import CNNModel, train_and_evaluate_model

def test_cnn_model_forward_shape():
    model = CNNModel((3, 60), filters=32, kernel_size=3, pooling_size=2, dropout_rate=0.3, activation='relu')
    dummy_input = torch.randn(8, 3, 60)  # batch_size = 8
    output = model(dummy_input)
    assert output.shape == (8, 1)

def test_training_loop_runs():
    # Generate dummy data
    X_train = np.random.rand(32, 60, 3)
    y_train = np.random.rand(32, 1)
    X_val = np.random.rand(10, 60, 3)
    y_val = np.random.rand(10, 1)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
            torch.tensor(y_train_scaled, dtype=torch.float32)
        ),
        batch_size=16, shuffle=False
    )

    model = CNNModel((3, 60), 32, 3, 2, 0.3, "relu")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    results = train_and_evaluate_model(
        model, train_loader, (X_val, y_val_scaled),
        optimizer, criterion, scaler_y,
        forecast_window=1, epochs=2
    )

    assert len(results) == 5
    assert all(isinstance(r, float) or np.isnan(r) for r in results)
