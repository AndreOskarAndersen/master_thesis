* 2/2:
    1. Data spørgsmål:
        1. Har også fundet andre datasæt - hvad med pretæning på disse og så finetune på ClimbAlong data?
        2. Måske kombinere flere datasæt?
    2. Model spørgsmål:
        1. Skal vi bare scratche den allerede implementerede Mask R-CNN eller skal vi udvide den?
        2. Skal jeg også inkorporere noget temporal-information i tracking af grips eller skal vi bare fokusere på pose estimation?
        3. Forskellige variationer af den samme model eller skal vi eksperimentere med forskellige modeller?
        4. Single- eller multi-person tracking?
        5. Nogle modeller gør brug af en 3D convolution istedet for en RNN - burde jeg også prøve dette eller skal vi bare holde os til RNN?
    3. Generelle spørgsmål:
        1. Hvor meget ML teori?
        2. Foreslå fremgangsmåde:
            1. Implementer temporal model
            2. Træn temporal model på et public dataset - kun på skeleterne (+ noise?)
            3. Finetune på ClimbAlong datasæt (+ noise?)
            4. Finetune hele R-CNN + temporal model på ClimbAlong?

* 8/2:
    1. Dataset:
        1. BRACE ser god ud synes jeg - nogle frames er dog ikke helt korrekt annoteret.

    2. Model:
        1. Mange modeller bruger ikke ResNet som backbone, men istedet HRNet - skal jeg også prøve dette?
        2. Unipose-LSTM ser stadig spændende ud
        3. DeciWatch ser også spændende ud (især fordi den er efficient)
        4. Skal jeg prøve både at lave en/to LSTM-modeller (Unipose) og en Transformer-model (DeciWatch)?

    3. Andre spørgsmål:
        1. Skal jeg køre med lavere precision (eks. 16-bit)?