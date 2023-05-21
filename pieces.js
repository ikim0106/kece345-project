const sharp = require('sharp')

sharp('./kangta.jpg')
    .extract({ left: 0, top: 0, width: 100, height: 100 })
    .toFile('./kangta.new.jpg', function (err) {
        if (err) console.log(err);
    })
