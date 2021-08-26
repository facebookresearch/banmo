import torch
import pdb


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )

class CondNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        use_viewdirs=True,
        codesize = 64,
        out_channel = 3,
        joint=False,
        time=False,
    ):
        super(CondNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        if joint:
            self.dim_xyz = include_input_xyz + 2 * num_encoding_fn_xyz**2 + codesize
            #self.dim_xyz = include_input_xyz + 2 * num_encoding_fn_xyz**3 + codesize
        elif time:
            self.dim_xyz = 1 + 2 * num_encoding_fn_xyz + codesize
        else:
            self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz + codesize
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if self.skip_connect_every is not None and i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        self.layers_dir = torch.nn.ModuleList()
        # This deviates from the original paper, and follows the code release instead.
        self.codesize=codesize
        self.layers_dir.append(
            torch.nn.Linear(hidden_size, hidden_size // 2)
        )
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, out_channel)
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        code=x[:,-self.codesize:]
        x=x[:,:-self.codesize]
        if self.use_viewdirs:
            xyz, view = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        else:
            xyz = x[..., : self.dim_xyz]
        xyz = torch.cat([xyz,code],1)
        x = self.layer1(xyz)
        for i in range(len(self.layers_xyz)):
            if (
                self.skip_connect_every is not None
                and i % self.skip_connect_every == 0
                and i > 0
                and i != len(self.layers_xyz) - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(self.layers_xyz[i](x))

        feat = self.relu(self.fc_feat(x))
        alpha = self.fc_alpha(x)
        x = feat
        for l in self.layers_dir:
            x = self.relu(l(x))
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)

