* 2/2:
    1. Torvalds datasæt er i 3D - kan jeg bare projektere det ned til 2D ved at fjerne en akse?
    2. Har også fundet andre datasæt
    3. Hvor mange tempoeral-arkitekturer skal jeg prøve?
    4. Skal jeg også inkorporere noget tempoeral-information i tracking af grips eller skal vi bare fokusere på pose estimation?
    5. Hvad synes han om UniPose?
    6. Nogle modeller gør brug af en 3D convolution istedet for en rnn
    7. Single- eller multi-person tracking? Tror kun UniPose er til single-person
    8. Pre-træne på større dataset også fine-tune på ClimbAlong-data?
    9. Foreslå fremgangsmåde:
        1. Implementer LSTM fra UniPose
        2. Træn LSTM på et public dataset - kun på skeleterne (+ noise?)
        3. Finetune på ClimbAlong datasæt (+ noise?)
        4. Finetune hele R-CNN + LSTM på ClimbAlong?