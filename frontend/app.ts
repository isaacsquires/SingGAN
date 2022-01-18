interface Response {
  img: string;
}

const imgElement = document.createElement("img");
document.querySelector("body").appendChild(imgElement);

function setImage(imgSrc: string) {
  imgElement.src = `data:image/png;base64, ${imgSrc}`;
}

async function postAudioData(audio: Float32Array, model: string) {
  try {
    const res = await fetch("/eval", {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        audio: Array.from(audio),
        model: model,
      }),
    });
    if (res.ok) {
      const { img }: Response = (await res.json()) as any;
      setImage(img);
    }
  } catch (error) {
    console.log(error);
  }
}

async function processAudio(stream: MediaStream) {
  const context = new AudioContext({
    latencyHint: undefined,
    sampleRate: 3000,
  });
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(256, 1, 1);

  source.connect(processor);
  processor.connect(context.destination);

  processor.onaudioprocess = async function (event) {
    const channelData = event.inputBuffer.getChannelData(0);
    const model = document.querySelector('input[name="model"]:checked').value;
    await postAudioData(channelData, model);
  };
}

function getLocalStream() {
  navigator.mediaDevices
    .getUserMedia({ video: false, audio: true })
    .then((stream) => {
      // Send audio data to backend
      processAudio(stream);
    })
    .catch((err) => {
      console.log("u got an error:" + err);
    });
}

getLocalStream();
