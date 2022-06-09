### Model train pipeline

...

### Data stracture


```bash
data
├── markup_opposite
│   ├── 0.png
│   ├── 1.png
│   └──...
├── markup_target
│   ├── 2.png
│   ├── 3.png
│   └──...
├── opposite
│   ├── 4.png
│   ├── 5.png
│   └──...
├── target
│   ├── 6.png
│   ├── 7.png
│   └──...
├── test_opposite
│   ├── 8.png
│   ├── 9.png
│   └──...
└── test_target
    ├── 10.png
    ├── 11.png
    └──...
```

Put raw photo to this, and then they will be parsed
```bash
markup
├── opposite
│   ├── 0.jpg
│   ├── 1.png
│   └──...
├── target
│   ├── 2.png
│   ├── 3.png
│   └──...
├── test_opposite
│   ├── 4.png
│   ├── 5.jpg
│   └──...
└── test_target
    ├── 6.jpg
    ├── 7.jpg
    └──...
```