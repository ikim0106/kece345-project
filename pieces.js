import fs from 'fs'
import sharp from 'sharp'
import csv from 'csv-parser'

const libFolder = './2D-Chessboard-and-Chess-Pieces-4'
let chessData = []
let filenames = []
let boardData = []

fs.createReadStream(libFolder+'/train/_annotations.csv')
    .pipe(csv())
    .on('data', (data) => {
        let csvFilename = data.filename
        let splits = csvFilename.split('.')[0].split('_')[2]
        // console.log(splits)
        if(!filenames.includes(csvFilename)) {
            filenames.push(csvFilename)
            chessData.push([csvFilename, splits])
        }
    })
    .on('end', () => {
        chessData.push([filenames, boardData])
        let nothing=0, whitePawn=0, blackPawn=0, whiteBishop=0, blackBishop=0, whiteKnight=0, blackKnight=0, whiteKing=0, blackKing=0, whiteQueen=0, blackQueen=0, whiteRook=0, blackRook = 0

        for(let i=0; i<500; i++) {
            let filename = chessData[i][0]
            let gameData = chessData[i][1].split('-')
            // console.log(filename, gameData)

            let leftMargin = 0
            let topMargin = 0


            for(let x=0; x<8; x++) {
                leftMargin=0
                for(let y=0; y<8; y++) {
                    let pieceType = ""
                    // console.log("leftMargin ", leftMargin, "topMargin ", topMargin, "gameData ", gameData[x][y])
                    switch (gameData[x][y]) {
                        case "1":
                            //nothing
                            pieceType = "nothing"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + nothing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            nothing++
                            break
                        case "R":
                            //white rook
                            pieceType = "whiterook"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + whiteRook + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteRook++
                            break
                        case "r":
                            //black rook
                            pieceType = "blackrook"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + blackRook + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackRook++
                            break
                        case "N":
                            //white knight
                            pieceType = "whiteknight"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + whiteKnight + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteKnight++
                            break
                        case "n":
                            //black knight
                            pieceType = "blackknight"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + blackKnight + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackKnight++
                            break
                        case "B":
                            //white bishop
                            pieceType = "whitebishop"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + whiteBishop + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteBishop++
                            break
                        case "b":
                            //black bishop
                            pieceType = "blackbishop"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + blackBishop + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackBishop++
                            break
                        case "Q":
                            //white queen
                            pieceType = "whitequeen"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + whiteQueen + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteQueen++
                            break
                        case "q":
                            //black queen
                            pieceType = "blackqueen"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + blackQueen + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackQueen++
                            break
                        case "P":
                            //white pawn
                            pieceType = "whitepawn"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + whitePawn + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whitePawn++
                            break
                        case "p":
                            //black pawn
                            pieceType = "blackpawn"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + blackPawn + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackPawn++
                            break
                        case "K":
                            //white king
                            pieceType = "whiteking"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + whiteKing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteKing++
                            break
                        case "k":
                            //black king
                            pieceType = "blackking"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./trainpieces/'+ pieceType + "/" + blackKing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackKing++
                            break
                    }
                    leftMargin += 52
                }
                topMargin += 52
            }
        }

        nothing=0, whitePawn=0, blackPawn=0, whiteBishop=0, blackBishop=0, whiteKnight=0, blackKnight=0, whiteKing=0, blackKing=0, whiteQueen=0, blackQueen=0, whiteRook=0, blackRook = 0

        for(let i=750; i<850; i++) {
            let filename = chessData[i][0]
            let gameData = chessData[i][1].split('-')
            // console.log(filename, gameData)

            let leftMargin = 0
            let topMargin = 0


            for(let x=0; x<8; x++) {
                leftMargin=0
                for(let y=0; y<8; y++) {
                    let pieceType = ""
                    // console.log("leftMargin ", leftMargin, "topMargin ", topMargin, "gameData ", gameData[x][y])
                    switch (gameData[x][y]) {
                        case "1":
                            //nothing
                            pieceType = "nothing"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + nothing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            nothing++
                            break
                        case "R":
                            //white rook
                            pieceType = "whiterook"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + whiteRook + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteRook++
                            break
                        case "r":
                            //black rook
                            pieceType = "blackrook"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + blackRook + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackRook++
                            break
                        case "N":
                            //white knight
                            pieceType = "whiteknight"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + whiteKnight + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteKnight++
                            break
                        case "n":
                            //black knight
                            pieceType = "blackknight"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + blackKnight + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackKnight++
                            break
                        case "B":
                            //white bishop
                            pieceType = "whitebishop"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + whiteBishop + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteBishop++
                            break
                        case "b":
                            //black bishop
                            pieceType = "blackbishop"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + blackBishop + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackBishop++
                            break
                        case "Q":
                            //white queen
                            pieceType = "whitequeen"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + whiteQueen + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteQueen++
                            break
                        case "q":
                            //black queen
                            pieceType = "blackqueen"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + blackQueen + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackQueen++
                            break
                        case "P":
                            //white pawn
                            pieceType = "whitepawn"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + whitePawn + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whitePawn++
                            break
                        case "p":
                            //black pawn
                            pieceType = "blackpawn"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + blackPawn + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackPawn++
                            break
                        case "K":
                            //white king
                            pieceType = "whiteking"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + whiteKing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteKing++
                            break
                        case "k":
                            //black king
                            pieceType = "blackking"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./validpieces/'+ pieceType + "/" + blackKing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackKing++
                            break
                    }
                    leftMargin += 52
                }
                topMargin += 52
            }
        }
        nothing=0, whitePawn=0, blackPawn=0, whiteBishop=0, blackBishop=0, whiteKnight=0, blackKnight=0, whiteKing=0, blackKing=0, whiteQueen=0, blackQueen=0, whiteRook=0, blackRook = 0

        for(let i=500; i<750; i++) {
            let filename = chessData[i][0]
            let gameData = chessData[i][1].split('-')
            // console.log(filename, gameData)

            let leftMargin = 0
            let topMargin = 0


            for(let x=0; x<8; x++) {
                leftMargin=0
                for(let y=0; y<8; y++) {
                    let pieceType = ""
                    // console.log("leftMargin ", leftMargin, "topMargin ", topMargin, "gameData ", gameData[x][y])
                    switch (gameData[x][y]) {
                        case "1":
                            //nothing
                            pieceType = "nothing"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + nothing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            nothing++
                            break
                        case "R":
                            //white rook
                            pieceType = "whiterook"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + whiteRook + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteRook++
                            break
                        case "r":
                            //black rook
                            pieceType = "blackrook"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + blackRook + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackRook++
                            break
                        case "N":
                            //white knight
                            pieceType = "whiteknight"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + whiteKnight + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteKnight++
                            break
                        case "n":
                            //black knight
                            pieceType = "blackknight"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + blackKnight + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackKnight++
                            break
                        case "B":
                            //white bishop
                            pieceType = "whitebishop"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + whiteBishop + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteBishop++
                            break
                        case "b":
                            //black bishop
                            pieceType = "blackbishop"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + blackBishop + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackBishop++
                            break
                        case "Q":
                            //white queen
                            pieceType = "whitequeen"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + whiteQueen + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteQueen++
                            break
                        case "q":
                            //black queen
                            pieceType = "blackqueen"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + blackQueen + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackQueen++
                            break
                        case "P":
                            //white pawn
                            pieceType = "whitepawn"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + whitePawn + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whitePawn++
                            break
                        case "p":
                            //black pawn
                            pieceType = "blackpawn"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + blackPawn + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackPawn++
                            break
                        case "K":
                            //white king
                            pieceType = "whiteking"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + whiteKing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            whiteKing++
                            break
                        case "k":
                            //black king
                            pieceType = "blackking"
                            sharp('./2D-Chessboard-and-Chess-Pieces-4/train/'+filename)
                                .extract({left:leftMargin, top:topMargin, width:52, height:52})
                                .toFile('./testpieces/'+ pieceType + "/" + blackKing + '.jpg', function(err) {
                                    if(err) console.log(err)
                                })
                            blackKing++
                            break
                    }
                    leftMargin += 52
                }
                topMargin += 52
            }
        }
    })
