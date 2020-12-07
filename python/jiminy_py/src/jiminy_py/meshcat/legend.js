function draggable(el) {
  el.style.cursor = "move";
  el.style.userSelect = "none";

  el.addEventListener("mousedown", function(e) {
    var offsetX = e.clientX - parseInt(window.getComputedStyle(this).left);
    var offsetY = e.clientY - parseInt(window.getComputedStyle(this).top);

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
  });
}

function createLegend(id) {
    legend = document.createElement("div");
    legend.id = id;
    legend.style.position = "fixed";
    legend.style.top = "20px";
    legend.style.left = "20px";
    legend.style.backgroundColor = "rgba(255, 255, 255, 0.9)";
    legend.style.border = "2px solid";
    legend.style.borderRadius = "5px";
    legend.style.borderColor = "Silver";
    legend.style.padding = "4px 10px 4px 8px";
    document.body.prepend(legend);

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
    label.innerHTML = text;
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
        legend = elem.parentNode
        legend.removeChild(elem);
        if (!legend.childElementCount)
        {
            document.body.removeChild(legend);
        }
    }
}
