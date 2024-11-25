# Simple ChatBot 
Simple ChatBot application based on NLU for basic conversation.

## Configuring the Bot
Changes need to be made to `intent.json` and `replies.json` based on user preferences.

### Example:
**File: intent.json**
`json
{
    "intent": [
        {
            "label": "greeting",
            "pattern": ["hello", "hi", "hey", "greetings", "how are you",
                        "hiya", "hey there", "hi there", "heya", "how do you do"]
        }
    ]
}`

File: replies.json:
`{
    "replies": [
        {
            "label": "greeting",
            "reply": [
                "Hello",
                "Hi",
                "Hey",
                "Hi there",
                "Hiya"
            ]
        },
}`

Each intent label should contains its corresponding replies.

## Usage
navigate to the root directory and follow the steps below:
## Training the Model

### Basic Training
Run the default training configuration:
```bash
python train.py
```

### Customizing Training Parameters
Customize training by specifying parameters:
`bash
python train.py \
    --epoch 50 \
    --learning_rate 0.001 \
    --batch_size 32 \
    --feature 128
`

#### Available Training Parameters
- `--epoch`: Number of training epochs (default: 1000)
- `--learning_rate`: Learning rate for optimization (default: 0.02)
- `--batch_size`: Number of samples per training batch (default: 16)
- `--feature`: Feature dimension size (default: None)

## Chatting with the Model
After training, interact with the model:
`bash
python chat.py
`
