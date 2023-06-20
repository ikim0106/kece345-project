const fs = require('fs')
const { MessageAttachment } = require('discord.js')
const https = require('https')
const { execSync, spawnSync } = require('node:child_process')
const util = require('util')
const sharp = require('sharp')
const { Chess } = require('chess.js')
sharp.cache(false)

let gameData = []

let download = function (url, dest, cb) {
    let file = fs.createWriteStream(dest)
    let request = https
        .get(url, function (response) {
            response.pipe(file)
            file.on('finish', function () {
                file.close(cb)
            });
        })
        .on('error', function (err) {
            fs.unlink(dest) // Delete the file async if there is an error
            if (cb) cb(err.message)
        });

    request.on('error', function (err) {
        console.log(err);
    });
};

exports.board = async function(message, client, args) {
    const iterator = message.attachments.entries()
    let boardImage = iterator.next().value
    console.log(args)

    let url = boardImage[1].attachment
    const imgpath = './cache/'+boardImage[1].id+'.jpg'
    download(url, imgpath, function (err) {
        if (err) {
            console.log(err)
        } else {
            execSync(`python imgengine.py ./cache/${boardImage[1].id}.jpg ${args[0]}`, {stdio: ['inherit', 'inherit', 'inherit']}, (error, stdout, stderr)=> {
                if (error) {
                    console.error(`exec error: ${error}`)
                    return
                }
                console.log(`stdout: ${stdout}`)
            })
            let data = fs.readFileSync('./cache/temp.svg', "utf-8", (err, data) => {
                console.log(data)
            })
            // console.log(data)
            // console.log('done')
            sharp('./cache/temp.svg')
                .png()
                .toFile(`./cache/${boardImage[1].id}-reply.png`)
                .then(function(info) {
                    message.reply({files: [`./cache/${boardImage[1].id}-reply.png`]})
                  })
                  .catch(function(err) {
                    console.log(err)
                  })
        }
    })
}

exports.playmove = async function(message, client, args) {
    let gameFEN = fs.readFileSync('./cache/FEN.txt', "utf-8", (err, data) => {
        if(err) console.log(err)
        else console.log(data)
    })

    console.log("gameFEN", gameFEN)
    // console.log(gameFEN)
    const chess = new Chess(gameFEN)
    if(chess.isCheckmate()) {
        message.reply('YOU LOSE')
        return
    }
    try{
        chess.move(args[0])
        // console.log(chessfen)
        
        // console.log(message.id)
        let chessfen = chess.fen()
        execSync(`python engine.py "${chessfen}"`, {stdio: ['inherit', 'inherit', 'inherit']}, (error, stdout, stderr)=> {
            if(error) console.log(error)
            else console.log(stdout)
        })
    }
    catch(e) {
        console.log(e)
        message.reply('Illegal move! (or unknown error)')
    }
    
    let data = fs.readFileSync('./cache/temp.svg', "utf-8", (err, data) => {
        console.log(data)
    })

    gameFEN = fs.readFileSync('./cache/FEN.txt', "utf-8", (err, data) => {
        if(err) console.log(err)
        else console.log(data)
    })

    if(gameFEN==="CHECKMATE") {
        message.reply('YOU WIN')
        return
    }
    
    sharp('./cache/temp.svg')
        .png()
        .toFile(`./cache/${message.id}-reply.png`)
        .then(function(info) {
            message.reply({files: [`./cache/${message.id}-reply.png`]})
        })
        .catch(function(err) {
            console.log(err)
        })

}