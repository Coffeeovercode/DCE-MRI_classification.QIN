import torch
import torch.nn as nn
import torchvision.models as models

class UNetHybrid(nn.Module):
    def __init__(self, encoder_name, in_channels=3, out_channels=1, pretrained=True):
        super(UNetHybrid, self).__init__()
        self.encoder, self.encoder_channels = self._get_encoder(encoder_name, pretrained)
        self.decoder = self._build_decoder()

        # Final convolution to map to output channels
        self.final_conv = nn.Conv2d(self.encoder_channels[0], out_channels, kernel_size=1)

    def _get_encoder(self, name, pretrained):
        if name == "resnet":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            channels = [64, 256, 512, 1024, 2048]
            encoder_layers = list(base_model.children())
            encoder = nn.Sequential(
                *encoder_layers[:3], # Conv, BN, ReLU
                *encoder_layers[4:8]  # Layer 1 to 4
            )
            return encoder, channels
        elif name == "mobilenet":
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None).features
            channels = [16, 24, 32, 96, 1280]
            return base_model, channels

        else:
            raise ValueError(f"Encoder '{name}' not supported.")

    def _build_decoder(self):
        decoder_layers = nn.ModuleList()
        # Reverse encoder channels for decoder path, skipping the last one (bottleneck)
        reversed_channels = self.encoder_channels[::-1]

        for i in range(len(reversed_channels) - 1):
            in_c = reversed_channels[i]
            out_c = reversed_channels[i+1]
            decoder_layers.append(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
            )
            decoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True)
                )
            )
        return decoder_layers

    def forward(self, x):
        skip_connections = []
        # Encoder path
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
        
        # We need the last output as the starting point for the decoder
        x = skip_connections.pop()
        skip_connections = skip_connections[::-1] # Reverse for decoder

        # Decoder path
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x) # Upsampling
            skip = skip_connections[i//2]
            
            if x.shape != skip.shape:
                # Pad if necessary
                x = torch.nn.functional.interpolate(x, size=skip.shape[2:])
            
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.decoder[i+1](concat_skip) # Double conv

        return torch.sigmoid(self.final_conv(x))
