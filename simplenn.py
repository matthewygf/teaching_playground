import torch

device = torch.device('cpu')

batch_size = 64
input_dim = 1000

# hidden , output
hidden_dim = 100
output_dim = 10

input_x = torch.randn(batch_size, input_dim, device=device)
output_y = torch.randn(batch_size, output_dim, device=device)

w1 = torch.randn(input_dim, hidden_dim, device=device)
w2 = torch.randn(hidden_dim, output_dim, device=device)

lr = 1e-6

for t in range(500):
    #forward 
    h = input_x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    loss = (y_pred - output_y).pow(2).sum()
    print(t, loss.item())

    # backprop
    grad_y_pred = 2.0 * (y_pred-output_y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_y_pred = grad_y_pred.mm(w2.t())
    grad_h = grad_y_pred.clone()
    grad_h[h<0] = 0
    grad_w1 = input_x.t().mm(grad_h)

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
