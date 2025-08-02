import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, config):
        super(DenseNet, self).__init__()
        self.config = config
        
        # Initialize model parameters from config
        self.num_classes = config.get('num_classes', 1000)
        self.growth_rate = config.get('growth_rate', 32)
        self.block_config = config.get('block_config', [6, 12, 24, 16])
        self.num_init_features = config.get('num_init_features', 64)
        self.drop_rate = config.get('drop_rate', 0.0)
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, self.num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = self._make_dense_block(num_layers, num_features)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * self.growth_rate
            
            if i != len(self.block_config) - 1:
                # Transition layer
                trans = self._make_transition(num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_dense_block(self, num_layers, num_input_features):
        layers = []
        for i in range(num_layers):
            layers.append(self._make_dense_layer(num_input_features + i * self.growth_rate))
        return nn.Sequential(*layers)
    
    def _make_dense_layer(self, num_input_features):
        return nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, 4 * self.growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * self.growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * self.growth_rate, self.growth_rate, kernel_size=3, padding=1, bias=False)
        )
    
    def _make_transition(self, num_input_features):
        return nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_input_features // 2, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features, inplace=True)
        out = torch.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
