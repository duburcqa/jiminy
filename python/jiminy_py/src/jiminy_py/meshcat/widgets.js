var widgets = document.getElementById("widgets");
var widgetsCanvas;
var widgetsNeedRender = true;

window.onresize = function(event) {
    widgetsNeedRender = true;
};

// ***************** Legend utilities ******************

function draggable(el) {
  el.style.cursor = "move";
  el.style.pointerEvents = "auto";
  el.style.userSelect = "none";

  el.addEventListener("mousedown", function(e) {
      var offsetX = e.clientX - parseInt(window.getComputedStyle(this).left, 10);
      var offsetY = e.clientY - parseInt(window.getComputedStyle(this).top, 10);

      function mouseMoveHandler(e) {
          el.style.top = (e.clientY - offsetY) + "px";
          el.style.left = (e.clientX - offsetX) + "px";
      }

      function reset() {
        window.removeEventListener("mousemove", mouseMoveHandler);
        window.removeEventListener("mouseup", reset);
      }

      window.addEventListener("mousemove", mouseMoveHandler);
      window.addEventListener("mouseup", reset);

      widgetsNeedRender = true;
  });
}

function createLegend(id) {
    var legend = document.createElement("div");
    legend.id = id;
    legend.style.position = "fixed";
    legend.style.top = "20px";
    legend.style.left = "20px";
    legend.style.backgroundColor = "rgba(255, 255, 255, 0.9)";
    legend.style.border = "2px solid";
    legend.style.borderRadius = "5px";
    legend.style.borderColor = "Silver";
    legend.style.padding = "4px 10px 4px 8px";
    widgets.prepend(legend);

    draggable(legend);
}

function setLegendItem(legend, id, text, color) {
    var box = document.createElement("div");
    box.style.display = "inline-block";
    box.style.height = "20px";
    box.style.width = "20px";
    box.style.margin = "4px 8px 4px 0px";
    box.style.backgroundColor = color;

    var label = document.createElement("span");
    label.textContent = text;
    label.style.fontFamily = "Dejavu Sans";
    label.style.textAlign = "center";
    label.style.alignItems = "center";

    var boxContainer = document.getElementById(id);
    if (boxContainer == null) {
        boxContainer = document.createElement("div");
        boxContainer.id = id;
        boxContainer.style.display = "flex";
        boxContainer.style.alignItems = "center";
        legend.appendChild(boxContainer);
    }
    while (boxContainer.firstChild) {
        boxContainer.removeChild(boxContainer.lastChild);
    }
    boxContainer.appendChild(box);
    boxContainer.appendChild(label);
}

function removeLegendItem(legend, id) {
    var elem = document.getElementById(id);
    if (elem !== null) {
        legend = elem.parentNode;
        legend.removeChild(elem);
        if (!legend.childElementCount)
        {
            widgets.removeChild(legend);
        }
    }
}

function updateLegend(cmd) {
    var legend = document.getElementById("legend");
    if (legend == null) {
        createLegend("legend");
        legend = document.getElementById("legend");
    }
    if (cmd.text) {
        setLegendItem(legend, cmd.id, cmd.text, cmd.color);
    } else {
        removeLegendItem(legend, cmd.id);
    }
    widgetsNeedRender = true;
}

// ***************** Logo utilities ******************

function createLogo(id) {
    var logo = document.createElement("img");
    logo.id = id;
    logo.draggable = false;
    logo.style.position = "fixed";
    logo.style.bottom = "20px";
    logo.style.left = "20px";
    widgets.prepend(logo);
}

function setLogo(logo, dataURL, width, height) {
    logo.setAttribute('src', 'data:image/png;base64,' + dataURL);
    logo.style.width = width.toString() + "px";
    logo.style.height = height.toString() + "px";
}

function removeLogo(logo) {
    widgets.removeChild(logo);
}

function updateLogo(cmd) {
    var logo = document.getElementById("logo");
    if (cmd.data) {
        if (logo == null) {
            createLogo("logo");
            logo = document.getElementById("logo");
        }
        setLogo(logo, cmd.data, cmd.width, cmd.height);
    } else {
        if (logo !== null) {
            removeLogo(logo);
        }
    }
    widgetsNeedRender = true;
}

// **************** Widgets utilities ******************

async function getWidgetsCanvas(viewer) {
    if (widgetsNeedRender) {
        viewer.camera.updateProjectionMatrix();
        viewer.renderer.render(viewer.scene, viewer.camera);
        widgetsCanvas = await html2canvas(widgets, {
            allowTaint: true,
            useCORS: true,
            backgroundColor: "rgba(0,0,0,0)",
            removeContainer: true
        });
        widgetsNeedRender = false;
    }
    return widgetsCanvas;
}

async function captureFrameAndWidgets(viewer) {
    var snapshotCanvas = document.createElement('canvas');
    var ctx = snapshotCanvas.getContext('2d');
    ctx.canvas.width = document.documentElement.clientWidth;
    ctx.canvas.height = document.documentElement.clientHeight;

    var widgetsCanvas = await getWidgetsCanvas(viewer);
    viewer.camera.updateProjectionMatrix();
    viewer.renderer.render(viewer.scene, viewer.camera);
    viewer.animator.after_render();
    viewer.needs_render = false;

    ctx.drawImage(viewer.renderer.domElement, 0, 0);
    ctx.drawImage(widgetsCanvas, 0, 0);

    return snapshotCanvas;
}
