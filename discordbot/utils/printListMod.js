//same as printlist but modified for flag

exports.printListMod = function (message, inputArray, noOfEntries) {
  let list = inputArray

  function returnFinalPrint() {
    let finalPrint = `**Flag Leaderboard**\n\`\`\``
    for(let i =0; i<inputArray.length; i++) {
      let pepe
      let pepelaf = `${list[i][0]}`
      let pad
      if (i<9) pad = 15-list[i][0].length
      else pad = 14-list[i][0].length
      for(let q = 0; q< pad; q++) pepelaf+= ' '
      pepelaf+= `${list[i][1]} pts`
      finalPrint += '' + `${i+1}` + '. ' + `${pepelaf}\n`

      // if(pepe>0&&pepe<100) finalPrint+=`          ${noOfEntries[i]} entries`
      // else if (pepe>100&&pepe<1000) finalPrint+=`         ${noOfEntries[i]} entries`
    }
    finalPrint+=`\`\`\``
    return finalPrint
  }
  let bigshit = returnFinalPrint()
  if(inputArray.length>60) {
    let stuff = bigshit.split('60. ')
    stuff[0]+=`\`\`\``
    let othershit = `\`\`\`60. ` + stuff[1]
    message.channel.send(stuff[0])
    message.channel.send(othershit)
  }
  else 
    message.channel.send(bigshit)
}